"""Finish a predecessor's mixing metadata, reusing independent index work."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from production.packed_index import index_shard, publish
from production.packed_mixture import mixture_candidate
from production.preparation_queue import verify_artifact


def get_json(client,spec):
    bucket,key=spec['uri'][5:].split('/',1)
    raw=client.get_object(Bucket=bucket,Key=key)['Body'].read()
    if spec.get('sha256') and hashlib.sha256(raw).hexdigest()!=spec['sha256']:
        raise ValueError('mixing input checksum differs')
    if spec.get('bytes')is not None and len(raw)!=spec['bytes']:
        raise ValueError('mixing input byte count differs')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def prepare_phase(client,plan,work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    report,source_sha=get_json(client,plan['source_report'])
    if plan.get('expected_input_ledger'):
        actual=report['artifacts']['selected.sqlite'];expected=plan['expected_input_ledger']
        if any(actual[field]!=expected[field]for field in ['uri','bytes','sha256']):
            raise ValueError('mixing phase differs from the inherited allocation ledger')
    if (report['plan_sha256']!=plan['source_plan_sha256']
            or report['totals']['content_tokens']!=plan['expected_content_tokens']
            or not report.get('all_curriculum_quotas_satisfied')
            or not report.get('physical_creation_finished')):
        raise ValueError('predecessor is not the completed physical phase')
    cached={}
    for spec in plan.get('cached_index_reports',[]):
        indexed,_=get_json(client,spec)
        if indexed.get('status')!='complete':raise ValueError('incomplete cached index job')
        for row in indexed['shards']:
            cached[(row['payload']['uri'],row['index']['sha256'])]=row
    indexed=[];missing=[]
    for row in report['shards']:
        stored=row if 'index_offsets'in row else cached.get((row['payload']['uri'],row['index']['sha256']))
        if stored:
            for field in ['sequences','leaf','stratum','payload','index','plan_sha256']:
                if stored[field]!=row[field]:raise ValueError('cached index source binding differs')
            verify_artifact(client,stored['index_offsets']);indexed.append(stored)
        else:missing.append(row)
    with ThreadPoolExecutor(max_workers=8)as pool:
        indexed.extend(pool.map(lambda row:index_shard(client,plan,row,work),missing))
    candidate=mixture_candidate(report,[{'status':'complete','shards':indexed}])
    path=work/'MIXTURE_CANDIDATE.json';path.write_text(json.dumps(candidate,indent=2)+'\n')
    spec=publish(client,plan['bucket'],path,plan['output_prefix']+'/MIXTURE_CANDIDATE.json')
    result={'status':'complete','source_report_sha256':source_sha,
        'source_plan_sha256':report['plan_sha256'],'content_tokens':candidate['content_tokens'],
        'cached_shards_reused':len(indexed)-len(missing),'newly_indexed_shards':len(missing),
        'mixing_candidate':spec,'native_training_accepted':False}
    path=work/'FINALIZED.json';path.write_text(json.dumps(result,indent=2)+'\n')
    publish(client,plan['bucket'],path,plan['output_prefix']+'/FINALIZED.json')
    return result
