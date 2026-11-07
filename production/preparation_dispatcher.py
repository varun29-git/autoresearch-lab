"""S3-backed CPU preparation DAG dispatcher, deployed externally as a Lambda.

One reserved concurrent invocation owns STATE.json. A persisted launch request
and EC2 client token make an interrupted launch idempotent. Workers terminate
on exit and each also receives an independent Scheduler termination deadline.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import time

from production.preparation_queue import object_spec, ready_tasks, sha, tranches, validate_completion, verify_artifact


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()


def resolved_task(task, state, bucket, now):
    task = copy.deepcopy(task)
    plan = task['plan']
    plan['deadline_unix'] = now+task['runtime_seconds']
    plan['jobs_path'] = '/opt/ci1-next/'+task['jobs_file']
    plan['jobs_sha256'] = task['jobs_sha256']
    if task['kind'] == 'textbook':
        plan['tokenizer_path'] = '/opt/ci1-next/tokenizer/tokenizer-65k-v2.json'
        plan['language_model_path'] = '/opt/ci1-next/lid.176.bin'
    else:
        prior = state[task['ledger_from']]
        if prior['status'] != 'complete':
            raise ValueError('allocation ledger predecessor is not complete')
        plan['exclusions_path'] = '/opt/ci1-next/training-exclusions.jsonl'
        plan['input_ledger'] = prior['artifacts']['selected.sqlite']
        plan['input_suffixes'] = prior['artifacts']['quota-suffix-reservations.jsonl']
        if task.get('fixed_selection_from'):
            fixed = state[task['fixed_selection_from']]
            if fixed['status'] != 'complete':
                raise ValueError('fixed selection is not complete')
            plan['fixed_selection'] = fixed['artifacts']['protected-records.jsonl']
            task['source_targets'] = plan['source_targets'] = fixed['effective_source_targets']
            if task.get('fixed_partition_sizes'):
                sizes = task['fixed_partition_sizes']
                number = task['fixed_partition_index']
                if type(number) is not int or not 0 <= number < len(sizes):
                    raise ValueError('invalid fixed partition index')
                full = fixed['effective_source_targets']
                parts = tranches(full, sizes)
                selected = {k:parts[number].get(k,0) for k in full}
                starts = {k:sum(part.get(k,0) for part in parts[:number]) for k in full}
                plan['fixed_total_source_targets'] = full
                plan['fixed_source_ranges'] = {k:[starts[k],starts[k]+selected[k]] for k in full}
                task['source_targets'] = plan['source_targets'] = selected
    if plan['bucket'] != bucket:
        raise ValueError('task bucket differs from queue bucket')
    return task


def bootstrap(queue, task, task_key, task_sha):
    # All interpolations are validated AWS names, integers, or hex digests.
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/.'
           for value in [queue['bucket'], queue['prefix'], task['id'], task_key,
                         queue['runtime']['uri'][5:]] for c in value):
        raise ValueError('unsafe bootstrap object name')
    for digest in [task_sha, queue['runtime']['sha256']]:
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid bootstrap SHA-256')
    return f'''#!/bin/bash
set -Eeuo pipefail
exec > >(tee /var/log/ci1-next.log) 2>&1
finish() {{
 aws s3 cp /var/log/ci1-next.log s3://{queue['bucket']}/{task['completion_prefix']}/worker.log --region us-east-1 || true
 shutdown -h now
}}
trap finish EXIT
shutdown -h +{max(1,math.ceil(task['runtime_seconds']/60))}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1
ulimit -n 8192
mkdir -p /opt/ci1-next /work/ci1
cd /opt/ci1-next
dnf -y install python3.11 python3.11-pip python3.11-devel gcc-c++ unzip
python3.11 -m pip install --disable-pip-version-check --no-cache-dir 'boto3==1.40.0' 'pyarrow==22.0.0' 'regex==2026.7.19' 'unicodedata2==16.0.0' 'numpy==1.26.4' 'pybind11==3.0.1' 'setuptools==80.9.0' 'fasttext-wheel==0.9.2'
aws s3 cp {queue['runtime']['uri']} runtime.zip --region us-east-1 --only-show-errors
echo '{queue['runtime']['sha256']}  runtime.zip' | sha256sum -c -
unzip -q runtime.zip
aws s3 cp s3://{queue['bucket']}/{task_key} TASK.json --region us-east-1 --only-show-errors
echo '{task_sha}  TASK.json' | sha256sum -c -
python3.11 -m pip install --no-build-isolation ./native/tokenizer65k
python3.11 -m unittest tests.test_document_packing tests.test_packing_worker tests.test_textbook_source tests.test_preparation_queue
python3.11 -u -m production.preparation_queue --task TASK.json --assets /opt/ci1-next --work /work/ci1/task
'''


class Dispatcher:
    def __init__(self, queue, clients):
        self.q=queue
        self.s3,self.ec2,self.scheduler,self.events=clients
        self.bucket=queue['bucket'];self.prefix=queue['prefix']
        self.state=self.get_json(self.prefix+'/STATE.json') or {'tasks':{},'status':'active'}
        self.tasks={t['id']:t for t in queue['tasks']}

    def get_json(self, key):
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket,Key=key)['Body'].read())
        except self.s3.exceptions.NoSuchKey:
            return None

    def save(self):
        self.state['updated_unix']=int(time.time())
        self.s3.put_object(Bucket=self.bucket,Key=self.prefix+'/STATE.json',Body=encoded(self.state))

    def verified_report(self, spec):
        b,k=spec['uri'][5:].split('/',1)
        if b != self.bucket:
            raise ValueError('completion report is outside queue bucket')
        raw=self.s3.get_object(Bucket=b,Key=k)['Body'].read()
        if len(raw)!=spec['bytes'] or sha(raw)!=spec['sha256']:
            raise ValueError('completion report checksum differs')
        return json.loads(raw)

    def reconcile(self):
        state=self.state['tasks']
        for task in self.q['external']:
            tid=task['id']
            if state.get(tid,{}).get('status') in ('complete','failed'):
                continue
            report=self.get_json(task['report_key'])
            if report is None:
                if int(time.time())>task['deadline_unix']+300:
                    state[tid]={'status':'failed','error':'predecessor deadline passed without complete materialization'}
                continue
            try:
                validate_completion(task,report,task['plan_sha256'])
                artifacts={name:object_spec(self.s3,self.bucket,task['output_prefix']+'/audit/'+name)
                           for name in ['selected.sqlite','quota-suffix-reservations.jsonl']}
                state[tid]={'status':'complete','artifacts':artifacts,'report_key':task['report_key']}
            except ValueError as exc:
                state[tid]={'status':'failed','error':str(exc)}
        for tid,record in list(state.items()):
            if tid not in self.tasks or record['status'] not in ('running','launching'):
                continue
            task=record['resolved_task']
            complete=self.get_json(task['completion_prefix']+'/COMPLETE.json')
            if complete:
                try:
                    if (complete.get('status')!='complete' or complete.get('task_id')!=tid
                            or complete.get('plan_sha256')!=record['expected_plan_sha256']):
                        raise ValueError('task completion identity differs')
                    report=self.verified_report(complete['report'])
                    if task['kind']=='textbook':
                        if (report.get('status')!='verified'
                                or report.get('source_sha256')!=task['source_sha256']
                                or report.get('encoder_identity',{}).get('tokenizer_sha256')!=task['plan']['tokenizer_sha256']
                                or not report.get('encoder_identity',{}).get('roundtrip_checked_per_document')):
                            raise ValueError('textbook receipt is not the pinned source/tokenizer')
                        total=sum(o['tokens'] for o in report['outputs'] if not o['validation'])
                        if total!=complete['train_tokens']:
                            raise ValueError('textbook completion token accounting differs')
                    else:
                        validate_completion(task,report,record['expected_plan_sha256'])
                        if complete['artifacts']!=report['artifacts']:
                            raise ValueError('completion artifact bindings differ')
                        for name in ['selected.sqlite','quota-suffix-reservations.jsonl']:
                            verify_artifact(self.s3,complete['artifacts'][name])
                    record.update(complete,status='complete',completed_unix=int(time.time()))
                except ValueError as exc:
                    record.update(status='failed',error=str(exc))
                # Completion is durable before forcing a stuck bootstrap to exit.
                if record.get('instance_id'):
                    self.ec2.terminate_instances(InstanceIds=[record['instance_id']])
                continue
            failed=self.get_json(task['completion_prefix']+'/FAILED.json')
            if failed:
                record.update(status='failed',error=failed.get('error','worker failed'))
                if record.get('instance_id'):
                    self.ec2.terminate_instances(InstanceIds=[record['instance_id']])
                continue
            if record.get('instance_id'):
                desc=self.ec2.describe_instances(InstanceIds=[record['instance_id']])
                phase=desc['Reservations'][0]['Instances'][0]['State']['Name']
                if phase in ('terminated','shutting-down','stopped','stopping'):
                    record.update(status='failed',error='worker stopped without a valid completion receipt')
            if int(time.time())>task['plan']['deadline_unix']+60 and record['status'] in ('running','launching'):
                if record.get('instance_id'):
                    self.ec2.terminate_instances(InstanceIds=[record['instance_id']])
                record.update(status='failed',error='hard task deadline exceeded')
        self.save()

    def launch(self, original):
        q=self.q;tid=original['id'];state=self.state['tasks']
        if original['instance_type'] not in q['allowed_instance_types']:
            raise ValueError('instance type is not CPU-allowlisted')
        if tid not in state:
            task=resolved_task(original,state,self.bucket,int(time.time()))
            body=encoded(task);digest=sha(body)
            key=self.prefix+'/resolved/'+tid+'/'+digest+'.json'
            self.s3.put_object(Bucket=self.bucket,Key=key,Body=body)
            state[tid]={'status':'launching','resolved_task':task,'task_key':key,
                'task_sha256':digest,'expected_plan_sha256':sha(encoded(task['plan']))}
            self.save()
        record=state[tid];task=record['resolved_task']
        if task['plan']['deadline_unix']<=time.time()+120:
            record.update(status='failed',error='persisted launch request expired')
            self.save();return
        tag=[{'Key':'Name','Value':q['run']+'-'+tid},{'Key':'Run','Value':q['fleet_tag']},
             {'Key':'PreparationCampaign','Value':q['run']},{'Key':'Project','Value':'Terminus'}]
        instance_id=record.get('instance_id')
        try:
            if not instance_id:
                args={'ImageId':q['ami'],'InstanceType':task['instance_type'],'MinCount':1,'MaxCount':1,
                    'ClientToken':hashlib.sha256((q['run']+'/'+tid).encode()).hexdigest(),
                    'IamInstanceProfile':{'Name':q['worker_profile']},'InstanceInitiatedShutdownBehavior':'terminate',
                    'MetadataOptions':{'HttpTokens':'required','HttpEndpoint':'enabled'},
                    'NetworkInterfaces':[{'DeviceIndex':0,'SubnetId':q['subnet'],'Groups':[q['security_group']],
                                          'AssociatePublicIpAddress':True}],
                    'BlockDeviceMappings':[{'DeviceName':'/dev/xvda','Ebs':{'VolumeSize':task['volume_gb'],
                        'VolumeType':'gp3','Iops':6000,'Throughput':500,'DeleteOnTermination':True,'Encrypted':True}}],
                    'TagSpecifications':[{'ResourceType':r,'Tags':tag} for r in ['instance','volume','network-interface']],
                    'UserData':bootstrap(q,task,record['task_key'],record['task_sha256'])}
                instance_id=self.ec2.run_instances(**args)['Instances'][0]['InstanceId']
                record['instance_id']=instance_id;self.save()
            name=q['run']+'-'+sha(tid.encode())[:16]+'-stop'
            when=datetime.fromtimestamp(task['plan']['deadline_unix'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            schedule={'Name':name,'ScheduleExpression':'at('+when+')','ScheduleExpressionTimezone':'UTC',
                'FlexibleTimeWindow':{'Mode':'OFF'},'ActionAfterCompletion':'DELETE',
                'Target':{'Arn':'arn:aws:scheduler:::aws-sdk:ec2:terminateInstances','RoleArn':q['deadline_role_arn'],
                    'Input':json.dumps({'InstanceIds':[instance_id]}),
                    'RetryPolicy':{'MaximumEventAgeInSeconds':3600,'MaximumRetryAttempts':10}}}
            try:
                self.scheduler.create_schedule(**schedule)
            except self.scheduler.exceptions.ConflictException:
                pass
            actual=self.scheduler.get_schedule(Name=name)
            if (actual['State']!='ENABLED' or actual['Target']['Input']!=schedule['Target']['Input']
                    or actual['ScheduleExpression']!=schedule['ScheduleExpression']):
                raise ValueError('independent termination deadline differs')
            record.update(status='running',schedule=name,launched_unix=int(time.time()))
        except Exception as exc:
            # A timed-out RunInstances request may have created a VM even when
            # the response did not reach us. Resolve the persisted client token
            # before declaring the launch failed, rather than orphaning it.
            if not instance_id:
                try:
                    response=self.ec2.describe_instances(Filters=[{'Name':'client-token',
                        'Values':[hashlib.sha256((q['run']+'/'+tid).encode()).hexdigest()]}])
                    ids=[i['InstanceId'] for group in response['Reservations'] for i in group['Instances']]
                    if ids:
                        self.ec2.terminate_instances(InstanceIds=ids)
                        record['ambiguous_launch_terminated']=ids
                except Exception as recovery_error:
                    # Retain the SAME idempotent launch request for the next
                    # dispatcher invocation; this is not a fresh paid attempt.
                    record.update(status='launching',error=str(exc),recovery_error=str(recovery_error))
                    self.save();return
            if instance_id:
                self.ec2.terminate_instances(InstanceIds=[instance_id])
            record.update(status='failed',error=type(exc).__name__+': '+str(exc),automatic_paid_retry=False)
        self.save()

    def run(self):
        self.reconcile()
        if self.q['expires_unix']<=time.time():
            self.state['status']='expired';self.save()
            self.events.disable_rule(Name=self.q['event_rule']);return self.state
        state=self.state['tasks']
        # Finish the same persisted EC2 request after an interrupted invocation.
        for tid,row in list(state.items()):
            if row['status']=='launching':
                self.launch(self.tasks[tid])
        active=self.ec2.describe_instances(Filters=[{'Name':'tag:Run','Values':[self.q['fleet_tag']]},
            {'Name':'instance-state-name','Values':['pending','running']}])
        instances=[i for group in active['Reservations'] for i in group['Instances']]
        running=sum(r['status'] in ('running','launching') for r in state.values())
        slots=max(0,min(self.q['max_parallel']-running,self.q['max_total_tagged_instances']-len(instances)))
        for task in ready_tasks(self.q['tasks'],state)[:slots]:
            self.launch(task)
        # Propagate permanent failure, so a blocked child is never silently retried.
        changed=True
        while changed:
            changed=False
            for task in self.q['tasks']:
                if task['id'] not in state and any(state.get(d,{}).get('status') in ('failed','blocked') for d in task['depends_on']):
                    state[task['id']]={'status':'blocked','error':'required predecessor failed'};changed=True
        if all(state.get(t['id'],{}).get('status') in ('complete','failed','blocked') for t in self.q['tasks']):
            self.state['status']='complete' if all(state[t['id']]['status']=='complete' for t in self.q['tasks']) else 'blocked'
            self.events.disable_rule(Name=self.q['event_rule'])
        self.save()
        return {'status':self.state['status'],'counts':{status:sum(r['status']==status for r in state.values())
                for status in ['complete','running','launching','failed','blocked']}}


def handler(event, context):
    import boto3
    s3=boto3.client('s3')
    raw=s3.get_object(Bucket=os.environ['QUEUE_BUCKET'],Key=os.environ['QUEUE_KEY'])['Body'].read()
    if sha(raw)!=os.environ['QUEUE_SHA256']:
        raise ValueError('queue definition changed after deployment')
    q=json.loads(raw)
    return Dispatcher(q,(s3,boto3.client('ec2'),boto3.client('scheduler'),boto3.client('events'))).run()
