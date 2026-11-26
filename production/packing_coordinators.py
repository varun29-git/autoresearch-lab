"""Independent fixed-source coordinators sharing immutable local inputs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from botocore.exceptions import ClientError

from production.document_packing import file_sha
from production.native_corpus import publish
from production.preparation_queue import validate_completion


def census(path,targets,accepted,charges):
    """Every positive reservation belongs to one source group or one pair."""
    import orjson
    counts=Counter();records=Counter();skipped=Counter();seen_pairs=set();order=[];previous=None
    opener=gzip.open if str(path).endswith('.gz')else open
    with opener(path,'rb')as stream:
        for line in stream:
            row=orjson.loads(line);leaf=row['leaf'];n=row['allocated_content_tokens']
            if leaf not in targets or type(n)is not int or n<=0:
                raise ValueError('unassigned source or invalid token count in fixed input')
            if leaf!=previous:
                if leaf in order:return None
                order.append(leaf);previous=leaf
            counts[leaf]+=n;sha=row['source_text_sha256']
            if sha in accepted:
                if sha in seen_pairs:raise ValueError('paired record repeats in the fixed input')
                seen_pairs.add(sha);skipped[leaf]+=n
            else:records[leaf]+=1
    if dict(counts)!={k:n for k,n in targets.items()if n!=0}:
        raise ValueError('parallel fixed input differs from exact source quotas')
    if seen_pairs!=accepted or dict(skipped)!=dict(charges):
        raise ValueError('parallel pair exclusions differ from qualified source charges')
    net={leaf:counts[leaf]-skipped[leaf]for leaf in order if counts[leaf]>skipped[leaf]}
    return order,net,records


def groups_for(net,records,workers):
    groups=[{}for _ in range(min(workers,len(net)))];loads=[0]*len(groups)
    weights={leaf:2*n+1500*records[leaf]for leaf,n in net.items()}
    for leaf in sorted(net,key=lambda k:(-weights[k],k)):
        i=min(range(len(groups)),key=lambda j:(loads[j],j))
        groups[i][leaf]=net[leaf];loads[i]+=weights[leaf]
    return groups


def run_child(plan_path,work,cancel):
    log=work.parent/'worker.log';process=None
    with log.open('wb')as out:
        try:
            process=subprocess.Popen([sys.executable,'-u','-m','production.fast_packing_worker',
                '--plan',str(plan_path),'--work',str(work)],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=json.loads(plan_path.read_text())['deadline_unix']
            while process.poll()is None:
                if cancel.wait(0.2)or time.time()>deadline:
                    raise RuntimeError('parallel materializer cancelled or exceeded deadline')
            if process.returncode:raise RuntimeError('parallel materializer failed: '+log.read_text()[-4000:])
            return json.loads((work/'MATERIALIZATION.json').read_text())
        finally:
            if process is not None and process.poll()is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def run_pool(worker,selection,paired,runner=None):
    plan=worker.plan;n=plan['materializer_processes']
    if type(n)is not int or not 2<=n<=4 or not worker.readonly_ledger:
        raise ValueError('parallel coordinators require 2–4 workers and an immutable ledger')
    if plan.get('fixed_source_ranges')or plan.get('fixed_source_scope'):return None
    accepted=paired.accepted if paired else set();charges=paired.charges if paired else Counter()
    scanned=census(selection,plan['source_targets'],accepted,charges)
    if scanned is None:return None
    order,net,records=scanned
    if len(net)<2:return None
    groups=groups_for(net,records,n)
    worker.finish_stratum(force=True)
    base_totals=Counter(worker.totals);base_shards=list(worker.receipts)
    cache={spec['uri']:{'path':str(path.resolve()),'sha256':spec['sha256'],'bytes':spec['bytes']}
        for spec,path in [(plan['input_ledger'],worker.work/'input-ledger.sqlite'),(plan['fixed_selection'],selection)]}
    jobs=[];root=worker.work/'coordinators';root.mkdir()
    for i,targets in enumerate(groups):
        directory=root/str(i);directory.mkdir();child=copy.deepcopy(plan)
        for key in ('translation_pairs','plan_sha256','predecessor_mixture_preparation'):
            child.pop(key,None)
        child.update(materializer_processes=1,source_targets=targets,source_fallbacks={},
            fixed_total_source_targets=targets,fixed_source_scope=list(targets),fixed_skip_source_shas=sorted(accepted),
            local_verified_inputs=cache,metadata_workers=1,packing_workers=2,packing_prefetch=4,
            io_workers=3,upload_workers=2,upload_prefetch=2,input_ram_mib=4096,metadata_ram_mib=1024,
            output_prefix=worker.prefix+'/coordinators/'+str(i))
        path=directory/'plan.json';path.write_text(json.dumps(child,sort_keys=True,indent=2)+'\n')
        jobs.append({'plan':child,'path':path,'work':directory/'data','sha256':file_sha(path)})
    worker.event('parallel_materialization_started',coordinators=len(jobs),source_groups=[list(j['plan']['source_targets'])for j in jobs])
    reports={};cancel=threading.Event();runner=runner or run_child
    with ThreadPoolExecutor(max_workers=len(jobs))as pool:
        pending={pool.submit(runner,j['path'],j['work'],cancel):i for i,j in enumerate(jobs)}
        try:
            while pending:
                done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    i=pending.pop(future);report=future.result();job=jobs[i]
                    validate_completion(job['plan'],report,job['sha256'])
                    if report['artifacts']['selected.sqlite']!=plan['input_ledger']:
                        raise ValueError('parallel worker changed the immutable ownership binding')
                    reports[i]=report
                totals=Counter(base_totals)
                for i,job in enumerate(jobs):
                    if i in reports:totals.update(reports[i]['totals']);continue
                    try:
                        progress=json.loads(worker.client.get_object(Bucket=worker.bucket,
                            Key=job['plan']['output_prefix']+'/progress.json')['Body'].read())
                        totals.update(progress.get('totals',{}))
                    except ClientError as exc:
                        if exc.response['Error']['Code'] not in ('NoSuchKey','404'):raise
                worker.totals=totals
                worker.event('parallel_packing',coordinators=len(jobs),completed_coordinators=len(reports))
                worker.deadline()
        except BaseException:
            cancel.set();raise
    by_leaf={};counts=Counter(charges);evidence=[]
    for i,job in enumerate(jobs):
        report=reports[i]
        for row in report['results']:counts[row['leaf']]+=row['selected']
        for shard in report['shards']:
            if shard['plan_sha256']!=job['sha256']:
                raise ValueError('parallel shard differs from its validated child plan')
            adopted=shard|{'plan_sha256':plan['plan_sha256'],
                'coordinator_plan_sha256':job['sha256'],'coordinator_receipt':shard['receipt']}
            adopted.pop('receipt')
            receipt=root/('adopted-'+hashlib.sha256(json.dumps(adopted,sort_keys=True).encode()).hexdigest()+'.json')
            receipt.write_text(json.dumps(adopted,indent=2)+'\n')
            adopted['receipt']=publish(worker.client,worker.bucket,receipt,worker.prefix+'/coordinator-receipts/'+receipt.name)
            by_leaf.setdefault(shard['leaf'],[]).append(adopted)
        spec=publish(worker.client,worker.bucket,job['work']/'MATERIALIZATION.json',
            worker.prefix+'/coordinator-reports/'+str(i)+'.json')
        evidence.append({'source_targets':job['plan']['source_targets'],'report':spec,'plan_sha256':job['sha256']})
        log=job['path'].parent/'worker.log'
        if log.exists():publish(worker.client,worker.bucket,log,worker.prefix+'/coordinator-logs/'+str(i)+'.log')
    if dict(counts)!={k:v for k,v in plan['source_targets'].items()if v!=0}:
        raise ValueError('parallel coordinator source totals differ')
    worker.receipts=base_shards+[shard for leaf in order for shard in by_leaf.get(leaf,[])]
    worker.totals=Counter(base_totals)
    for report in reports.values():worker.totals.update(report['totals'])
    worker.coordinator_report={'workers':len(jobs),'source_groups':evidence,
        'input_ledger_sha256':plan['input_ledger']['sha256'],'fixed_selection_sha256':plan['fixed_selection']['sha256'],
        'qualified_pairs_removed_once':True,'all_source_quotas_reconciled':True}
    return [{'leaf':leaf,'target':target,'selected':counts[leaf],'shortfall':target-counts[leaf]}
        for leaf,target in plan['source_targets'].items()]
