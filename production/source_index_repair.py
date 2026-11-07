"""Validate and publish immutable, split-local source indices without rewriting bins.

Plans, journals, caches and summaries must be external to the production repo.
An index acceptance result does not certify corpus encoding or training quality.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

try:
    import orjson
except ImportError:
    orjson = None


def repair_index(raw, token_bytes, *, validation, modulus=1000):
    """Prove counts and split membership, then change only token_start values."""
    if isinstance(token_bytes, bool) or token_bytes < 0 or token_bytes % 2:
        raise ValueError('token bin must have a nonnegative even byte length')
    cursor = docs = changed = 0
    previous_end = 0
    output = bytearray()
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = orjson.loads(line) if orjson else json.loads(line)
        if not isinstance(row, dict):
            raise ValueError('index record is not an object')
        uid = row.get('doc_id')
        n = row.get('n_tokens')
        start = row.get('token_start')
        if not isinstance(uid, (str, int)) or isinstance(uid, bool) or uid == '':
            raise ValueError('document identity is absent or invalid')
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError('document token count is not a positive integer')
        if not isinstance(start, int) or isinstance(start, bool) or start < previous_end:
            raise ValueError('original index has invalid or overlapping document offsets')
        is_val = int.from_bytes(hashlib.blake2b(str(uid).encode(), digest_size=8).digest(), 'little') % modulus == 0
        if is_val != validation:
            raise ValueError('document hash split disagrees with the indexed token bin')
        previous_end = start + n
        changed += start != cursor
        row['token_start'] = cursor
        cursor += n
        if cursor * 2 > token_bytes:
            raise ValueError('index counts exceed the token bin')
        if orjson:
            output.extend(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
        else:
            output.extend(json.dumps(row, separators=(',', ':'), ensure_ascii=False, sort_keys=True).encode())
        output.append(10)
        docs += 1
    if cursor * 2 != token_bytes:
        raise ValueError('index counts do not exhaust the token bin; refusing to invent or truncate tokens')
    return bytes(output), dict(documents=docs, content_tokens=cursor, corrected_offsets=changed)


def s3_path(uri):
    if not uri.startswith('s3://'):
        raise ValueError('expected an S3 URI')
    bucket, key = uri[5:].split('/', 1)
    return bucket, key


def repair_object(client, task, output_prefix, cache=None):
    before_bucket, before_key = s3_path(task['tokens_uri'])
    index_bucket, index_key = s3_path(task['index_uri'])
    before = client.head_object(Bucket=before_bucket, Key=before_key)
    if (before['ETag'], before['ContentLength']) != (task['token_etag'], task['token_bytes']):
        raise ValueError('token object differs from the frozen repair plan')
    index_head = client.head_object(Bucket=index_bucket, Key=index_key)
    response = client.get_object(Bucket=index_bucket, Key=index_key, IfMatch=index_head['ETag'])
    raw = response['Body'].read()
    if len(raw) != index_head['ContentLength']:
        raise ValueError('index download length mismatch')
    etag = index_head['ETag'].strip('"')
    if re.fullmatch('[0-9a-f]{32}', etag) and hashlib.md5(raw).hexdigest() != etag:
        raise ValueError('index download does not match its single-part S3 ETag')
    original_sha = hashlib.sha256(raw).hexdigest()
    corrected, stats = repair_index(raw, before['ContentLength'], validation=task['validation'])
    corrected_sha = hashlib.sha256(corrected).hexdigest()
    prefix_bucket, prefix_key = s3_path(output_prefix.rstrip('/') + '/')
    destination = prefix_key + hashlib.sha256(task['index_uri'].encode()).hexdigest() + '/' + original_sha + '-' + corrected_sha + '.index.jsonl'
    checksum = base64.b64encode(hashlib.sha256(corrected).digest()).decode()
    metadata = {'original-index-sha256': original_sha, 'corrected-index-sha256': corrected_sha,
                'token-object-etag': before['ETag'].strip('"')}
    from botocore.exceptions import ClientError
    try:
        client.put_object(Bucket=prefix_bucket, Key=destination, Body=corrected,
                          ContentType='application/x-ndjson', Metadata=metadata,
                          ChecksumSHA256=checksum, IfNoneMatch='*')
    except ClientError as exc:
        if exc.response['Error']['Code'] not in ('PreconditionFailed', '412'):
            raise
    published = client.head_object(Bucket=prefix_bucket, Key=destination, ChecksumMode='ENABLED')
    if published['ContentLength'] != len(corrected) or published.get('Metadata') != metadata:
        raise ValueError('immutable published index metadata mismatch')
    if published.get('ChecksumSHA256') != checksum:
        raise ValueError('published index SHA-256 checksum mismatch')
    after = client.head_object(Bucket=before_bucket, Key=before_key)
    if (after['ETag'], after['ContentLength']) != (before['ETag'], before['ContentLength']):
        raise ValueError('token object changed during repair; published index is not accepted')
    cache_path = None
    if cache is not None:
        cache_path = cache / (hashlib.sha256(task['index_uri'].encode()).hexdigest() + '.index.jsonl.gz')
        with gzip.open(cache_path, 'wb', compresslevel=1) as handle:
            handle.write(corrected)
    return dict(status='index_verified', tokens_uri=task['tokens_uri'], original_index_uri=task['index_uri'],
                repaired_index_uri=f's3://{prefix_bucket}/{destination}', original_index_etag=index_head['ETag'],
                original_index_sha256=original_sha, repaired_index_sha256=corrected_sha,
                token_etag=before['ETag'], token_bytes=before['ContentLength'], validation=task['validation'],
                source=task['source'], **stats, local_gzip_cache=str(cache_path) if cache_path else None,
                token_payload_modified=False, complete_token_payload_hash_verified=False,
                corpus_encoder_acceptance=False)


def run(plan, output, profile, workers):
    import boto3
    from botocore.config import Config
    tasks = plan['objects']
    if len({t['index_uri'] for t in tasks}) != len(tasks) or len({t['tokens_uri'] for t in tasks}) != len(tasks):
        raise ValueError('repair plan repeats an index or token object')
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'indices-gzip'
    cache.mkdir(exist_ok=True)
    journal = output / 'journal.jsonl'
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            if r['status'] == 'index_verified':
                done[r['original_index_uri']] = r
    pending = [t for t in tasks if t['index_uri'] not in done or
               (t['token_etag'], t['token_bytes']) != (done[t['index_uri']]['token_etag'], done[t['index_uri']]['token_bytes'])]
    client = boto3.Session(profile_name=profile).client('s3', region_name='us-east-1', config=Config(
        max_pool_connections=workers*3, connect_timeout=10, read_timeout=60,
        retries={'mode':'standard','max_attempts':3}))
    started = time.monotonic()
    failures = []
    print(json.dumps({'objects':len(tasks),'previously_verified':len(tasks)-len(pending),'pending':len(pending)}), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, journal.open('a') as handle:
        futures = {pool.submit(repair_object, client, t, plan['immutable_output_prefix'], cache):t for t in pending}
        for i, future in enumerate(as_completed(futures), 1):
            t = futures[future]
            try:
                result = future.result()
                done[t['index_uri']] = result
            except Exception as exc:
                result = dict(status='rejected', original_index_uri=t['index_uri'], tokens_uri=t['tokens_uri'],
                              source=t['source'], error=type(exc).__name__+': '+str(exc))
                failures.append(result)
            handle.write(json.dumps(result)+'\n')
            handle.flush()
            if i % 100 == 0 or result['status'] != 'index_verified':
                print(json.dumps({'processed':i,'pending_total':len(pending),'verified_total':len(done),
                                  'rejected_this_pass':len(failures),'elapsed_seconds':round(time.monotonic()-started,1)}), flush=True)
    current = [done[t['index_uri']] for t in tasks if t['index_uri'] in done and
               (done[t['index_uri']]['token_etag'], done[t['index_uri']]['token_bytes']) == (t['token_etag'],t['token_bytes'])]
    summary = dict(planned_index_objects=len(tasks), verified_index_objects=len(current),
                   incomplete_index_objects=len(tasks)-len(current), rejected_this_pass=len(failures),
                   verified_documents=sum(x['documents'] for x in current),
                   corrected_offsets=sum(x['corrected_offsets'] for x in current),
                   content_tokens_by_source_and_split={}, token_payloads_modified=False,
                   full_token_payload_hash_audit=False, corpus_encoder_acceptance=False,
                   paid_compute_launched=any(x.get('runtime') == 'aws_lambda_cpu' for x in current),
                   paid_gpu_campaign_launched=False)
    for row in current:
        key = row['source'] + ('/validation' if row['validation'] else '/train')
        summary['content_tokens_by_source_and_split'][key] = summary['content_tokens_by_source_and_split'].get(key,0)+row['content_tokens']
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'verified-indices.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in current))
    print(json.dumps(summary), flush=True)
    return 0 if len(current)==len(tasks) else 2


def main():
    from production.artifacts import external_path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    if not 1<=args.workers<=32:
        p.error('workers must be between 1 and 32')
    plan = json.loads(external_path(args.plan).read_text())
    raise SystemExit(run(plan, external_path(args.output), args.profile, args.workers))


if __name__ == '__main__':
    main()
