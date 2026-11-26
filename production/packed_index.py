"""Build verified random-access descriptor indexes without selecting records.

This step can run before an allocation ledger is complete. It never changes
payloads, quotas, document ownership or heldout membership.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def descriptor_offsets(chunks, output, *, expected_bytes, expected_sha256, sequences):
    digest=hashlib.sha256();position=0;rows=0;last=None
    output.write(np.asarray([0],dtype='<u8').tobytes())
    for chunk in chunks:
        digest.update(chunk)
        ends=np.flatnonzero(np.frombuffer(chunk,dtype=np.uint8)==10)
        offsets=(ends.astype(np.uint64)+position+1).astype('<u8')
        output.write(offsets.tobytes());rows+=len(ends);position+=len(chunk)
        if chunk:last=chunk[-1]
        if position>expected_bytes:raise ValueError('descriptor exceeds pinned size')
    if (position!=expected_bytes or digest.hexdigest()!=expected_sha256
            or rows!=sequences or last!=10):
        raise ValueError('descriptor identity, row count or terminal newline differs')
    return rows


def publish(client,bucket,path,key):
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    checksum=base64.b64encode(bytes.fromhex(digest)).decode()
    with Path(path).open('rb')as body:
        client.put_object(Bucket=bucket,Key=key,Body=body,ChecksumSHA256=checksum,
                          Metadata={'sha256':digest})
    head=client.head_object(Bucket=bucket,Key=key,ChecksumMode='ENABLED')
    if head['ContentLength']!=Path(path).stat().st_size or head.get('ChecksumSHA256')!=checksum:
        raise ValueError('published descriptor offsets differ')
    return {'uri':'s3://'+bucket+'/'+key,'sha256':digest,'bytes':head['ContentLength']}


def index_shard(client,plan,shard,work):
    if shard.get('plan_sha256')!=plan['source_plan_sha256']:
        raise ValueError('packing plan identity differs')
    spec=shard['index'];bucket,key=spec['uri'][5:].split('/',1)
    if bucket!=plan['bucket']:raise ValueError('source bucket differs')
    path=Path(work)/(spec['sha256']+'.offsets.uint64')
    body=client.get_object(Bucket=bucket,Key=key)['Body']
    try:
        with path.open('wb',buffering=8<<20)as out:
            descriptor_offsets(iter(lambda:body.read(8<<20),b''),out,
                expected_bytes=spec['bytes'],expected_sha256=spec['sha256'],sequences=shard['sequences'])
    except BaseException:
        path.unlink(missing_ok=True);raise
    finally:body.close()
    result=publish(client,plan['bucket'],path,
        plan['output_prefix']+'/offsets/'+spec['sha256']+'.uint64')
    path.unlink()
    return shard|{'index_offsets':result,'index_offsets_bind_to_sha256':spec['sha256']}


def run(plan,work,client=None):
    if client is None:
        import boto3
        from botocore.config import Config
        client=boto3.client('s3',region_name='us-east-1',config=Config(max_pool_connections=24))
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    raw=Path(plan['jobs_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=plan['jobs_sha256']:raise ValueError('index job manifest differs')
    jobs=[json.loads(line)for line in raw.splitlines()]
    rows=[]
    with ThreadPoolExecutor(max_workers=plan.get('io_workers',8))as pool:
        futures={pool.submit(index_shard,client,plan,j,work):n for n,j in enumerate(jobs)}
        for future in as_completed(futures):
            if time.time()>plan['deadline_unix']-60:raise TimeoutError('descriptor index deadline')
            rows.append((futures[future],future.result()))
            print(json.dumps({'event':'descriptor_indexed','completed':len(rows),'assigned':len(jobs)}),flush=True)
    report={'schema':'ci1_descriptor_index_preparation_v1','status':'complete',
        'source_plan_sha256':plan['source_plan_sha256'],
        'jobs_sha256':plan['jobs_sha256'],'shards':[row for _,row in sorted(rows)],
        'total_sequences':sum(j['sequences']for j in jobs),
        'content_tokens':sum(j['content_tokens']for j in jobs),
        'changes_token_payloads':False,'changes_record_allocation':False,
        'training_manifest_published':False,'completed_unix':int(time.time())}
    path=work/'INDEXED.json';path.write_text(json.dumps(report,indent=2)+'\n')
    publish(client,plan['bucket'],path,plan['output_prefix']+'/INDEXED.json')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--work',required=True)
    a=p.parse_args();run(json.loads(Path(a.plan).read_text()),a.work)


if __name__=='__main__':main()
