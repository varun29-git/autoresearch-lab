"""Dependency and accounting rules for the external CPU preparation queue."""
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import json


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def split_integer(total, weights):
    denominator = sum(weights.values())
    if total < 0 or denominator <= 0 or any(v < 0 for v in weights.values()):
        raise ValueError('invalid quota weights')
    result = {k: total*v//denominator for k,v in weights.items()}
    ranked = sorted(weights,key=lambda k:(-(total*weights[k]%denominator),k))
    for key in ranked[:total-sum(result.values())]:
        result[key] += 1
    return result


def tranches(quotas, sizes):
    if sum(quotas.values()) != sum(sizes) or any(x <= 0 for x in sizes):
        raise ValueError('tranches do not exhaust assigned source budget')
    remaining = dict(quotas)
    result = []
    for size in sizes:
        by_language = defaultdict(dict)
        for source,n in remaining.items():
            by_language[source.split('/')[0]][source] = n
        language_targets = split_integer(size,{l:sum(v.values()) for l,v in by_language.items()})
        allocation = {}
        for language,target in language_targets.items():
            weights = by_language[language]
            allocation.update(split_integer(target,weights) if sum(weights.values()) else {k:0 for k in weights})
        for key,n in allocation.items():
            if n > remaining[key]:
                raise ValueError('tranche exceeds source remainder')
            remaining[key] -= n
        result.append({k:v for k,v in allocation.items() if v})
    if any(remaining.values()):
        raise ValueError('tranche rounding left unallocated source tokens')
    return result


def object_spec(client, bucket, key):
    head = client.head_object(Bucket=bucket,Key=key,ChecksumMode='ENABLED')
    checksum = head.get('ChecksumSHA256','')
    digest = head.get('Metadata',{}).get('sha256')
    if not digest and checksum and '-' not in checksum:
        digest = base64.b64decode(checksum,validate=True).hex()
    if not digest or len(digest) != 64:
        raise ValueError('dependency object has no full-file SHA-256')
    return {'uri':'s3://'+bucket+'/'+key,'sha256':digest,'bytes':head['ContentLength']}


def verify_artifact(client, spec):
    """Compare full object identity, retaining optional multipart evidence."""
    bucket,key = spec['uri'][5:].split('/',1)
    actual = object_spec(client,bucket,key)
    if any(spec.get(k) != value for k,value in actual.items()):
        raise ValueError('dependency artifact identity differs')
    return True


def validate_completion(task, report, expected_plan_sha=None):
    if not report.get('all_curriculum_quotas_satisfied'):
        raise ValueError('predecessor has unresolved source deficits')
    if expected_plan_sha and report.get('plan_sha256') != expected_plan_sha:
        raise ValueError('predecessor plan identity differs')
    selected = sum(r['selected'] for r in report['results'])
    if selected != sum(task['source_targets'].values()):
        raise ValueError('predecessor content budget differs')
    actual = defaultdict(int)
    for row in report['results']:
        if row['shortfall'] or row['selected'] != row['target']:
            raise ValueError('predecessor has inconsistent source accounting')
        actual[row['leaf']] += row['selected']
    expected = dict(task['source_targets'])
    for source,replacement in task.get('source_fallbacks',{}).items():
        shift = expected[source]-actual[source]
        if shift < 0:
            raise ValueError('fallback increased the donor source')
        expected[source] -= shift
        expected[replacement] += shift
    if dict(actual) != expected:
        raise ValueError('predecessor source mix differs outside authorized fallback')
    if task.get('reservation_only'):
        if not report.get('reservation_creation_finished') or report['totals'].get('reserved_content_tokens') != selected:
            raise ValueError('protected source reservation is incomplete')
    elif not report.get('physical_creation_finished') or report['totals'].get('content_tokens') != selected:
        raise ValueError('physical payload totals differ from source allocation')
    return True


def ready_tasks(tasks, state):
    complete = {k for k,v in state.items() if v.get('status')=='complete'}
    return [t for t in tasks if state.get(t['id'],{}).get('status','pending')=='pending'
            and set(t.get('depends_on',[])) <= complete]


def queue_runner(task_path, assets, work):
    """Run one pinned task; the instance bootstrap handles termination."""
    from pathlib import Path
    import subprocess
    import sys
    import boto3
    from production.document_packing import file_sha
    from production.native_corpus import publish
    root=Path(assets);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    task=json.loads(Path(task_path).read_text());client=boto3.client('s3',region_name='us-east-1')
    plan=task['plan']
    plan['jobs_path']=str(root/task['jobs_file'])
    plan['jobs_sha256']=file_sha(plan['jobs_path'])
    prefix=task['completion_prefix'];bucket=plan['bucket']
    try:
        if task['kind']=='textbook':
            plan.update(tokenizer_path=str(root/'tokenizer/tokenizer-65k-v2.json'),
                        language_model_path=str(root/'lid.176.bin'))
            module='production.textbook_source'
        else:
            plan.update(exclusions_path=str(root/'training-exclusions.jsonl'))
            module=plan.get('worker_module','production.packing_worker')
            if module not in {'production.packing_worker','production.fast_packing_worker'}:
                raise ValueError('unknown packing worker module')
        plan_path=work/'plan.json';plan_path.write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
        subprocess.run([sys.executable,'-u','-m',module,'--plan',str(plan_path),'--work',str(work/'data')],check=True)
        if task['kind']=='textbook':
            job=json.loads(Path(plan['jobs_path']).read_text())
            key=plan['output_prefix']+'/progress/'+job['job_id']+'.json'
            raw=client.get_object(Bucket=bucket,Key=key)['Body'].read();report=json.loads(raw)
            if report.get('status')!='verified' or report['source_sha256']!=job['source_sha256']:
                raise ValueError('textbook source did not produce a matching verified receipt')
            result={'status':'complete','task_id':task['id'],'report':{'uri':'s3://'+bucket+'/'+key,'sha256':sha(raw),'bytes':len(raw)},
                'train_tokens':sum(o['tokens'] for o in report['outputs'] if not o['validation'])}
        else:
            path=work/'data/MATERIALIZATION.json';report=json.loads(path.read_text())
            validate_completion(task,report,file_sha(plan_path))
            result={'status':'complete','task_id':task['id'], 'artifacts':report['artifacts'],
                'report':{'uri':'s3://'+bucket+'/'+plan['output_prefix']+'/MATERIALIZATION.json',
                          'sha256':file_sha(path),'bytes':path.stat().st_size},
                'effective_source_targets':{}}
            for row in report['results']:
                key=row['leaf'];result['effective_source_targets'][key]=result['effective_source_targets'].get(key,0)+row['selected']
        result['plan_sha256']=file_sha(plan_path)
        path=work/'COMPLETE.json';path.write_text(json.dumps(result,indent=2)+'\n')
        publish(client,bucket,path,prefix+'/COMPLETE.json')
    except BaseException as exc:
        path=work/'FAILED.json';path.write_text(json.dumps({'task_id':task['id'],'error':type(exc).__name__+': '+str(exc),
            'automatic_paid_retry':False},indent=2)+'\n')
        publish(client,bucket,path,prefix+'/FAILED.json')
        raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--assets',required=True);p.add_argument('--work',required=True)
    a=p.parse_args();queue_runner(a.task,a.assets,a.work)
