"""Pinned UltraData textbook ingestion for an external native-tokenizer plan."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

import boto3
import pyarrow.parquet as pq

from production.document_packing import file_sha
from production.native_corpus import publish
from terminus.tokenizer_native import Native65kTokenizer
from terminus.tokenizer_parallel import native_batch_pool


def english_eligible(text, classifier):
    # Classification input only: keep the actual training text unchanged.
    probe = ' '.join(text.split())
    labels, scores = classifier.predict(probe[:32768], k=1)
    if not labels or labels[0] != '__label__en' or float(scores[0]) < 0.8:
        return False
    # A long Chinese suffix must not be accepted through an English prefix.
    if len(probe) > 32768:
        labels, scores = classifier.predict(probe[-32768:], k=1)
        if not labels or labels[0] != '__label__en' or float(scores[0]) < 0.8:
            return False
    return True


def convert(job, plan, client, work, classifier, encoder, identity):
    work.mkdir(parents=True, exist_ok=True)
    source = work/'source.parquet'
    digest = hashlib.sha256()
    with urlopen(job['source_url'], timeout=90) as response, source.open('wb') as stream:
        while block := response.read(8 << 20):
            digest.update(block)
            stream.write(block)
    if digest.hexdigest() != job['source_sha256'] or source.stat().st_size != job['source_bytes']:
        raise ValueError('pinned textbook source hash or size changed')
    states = {}
    for split in (False, True):
        name = 'validation' if split else 'train'
        states[split] = dict(bin_path=work/(name+'.uint16'), index_path=work/(name+'.index.jsonl'),
                             tokens=0, documents=0)
        states[split]['bin'] = states[split]['bin_path'].open('wb')
        states[split]['index'] = states[split]['index_path'].open('w', encoding='utf-8')
    counts = Counter()
    row_number = 0
    next_report = 0
    prefix = plan['output_prefix']
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=512, columns=['uid', 'content']):
            if time.time() > plan['deadline_unix']-180:
                raise TimeoutError('textbook preparation deadline')
            selected, texts = [], []
            for uid, text in zip(batch.column('uid').to_pylist(), batch.column('content').to_pylist()):
                number = row_number
                row_number += 1
                counts['source_rows'] += 1
                if not isinstance(text, str) or not text.strip() or not isinstance(uid, str) or not uid:
                    counts['invalid_rows'] += 1
                    continue
                if not english_eligible(text, classifier):
                    counts['language_rejected_rows'] += 1
                    continue
                sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
                validation = int(hashlib.sha256((sha+':ultradata-textbooks-v2').encode()).hexdigest(),16)%10000 == 0
                selected.append((uid, number, sha, validation))
                texts.append(text)
            for (uid, number, sha, validation), data in zip(selected, encoder.encode_batch_bytes(texts)):
                state = states[validation]
                n = len(data)//2
                record = dict(doc_id=uid, row_key=job['source_key']+':'+str(number),
                    source_text_sha256=sha, split_family_id=uid, validation=validation,
                    token_start=state['tokens'], n_tokens=n)
                state['bin'].write(data)
                state['index'].write(json.dumps(record, ensure_ascii=False, separators=(',', ':'))+'\n')
                state['tokens'] += n
                state['documents'] += 1
            if time.monotonic() > next_report:
                progress = dict(status='encoding', job_id=job['job_id'], counts=dict(counts),
                    train_tokens=states[False]['tokens'], validation_tokens=states[True]['tokens'])
                client.put_object(Bucket=plan['bucket'], Key=prefix+'/progress/'+job['job_id']+'.json',
                                  Body=json.dumps(progress).encode())
                print(json.dumps(progress), flush=True)
                next_report = time.monotonic()+30
    finally:
        for state in states.values():
            state['bin'].close()
            state['index'].close()
    outputs = []
    for split, state in states.items():
        if state['documents']:
            key = prefix+'/outputs/'+job['job_id']+'/'+('validation' if split else 'train')
            outputs.append(dict(validation=split, documents=state['documents'], tokens=state['tokens'],
                bin=publish(client, plan['bucket'], state['bin_path'], key+'.uint16'),
                index=publish(client, plan['bucket'], state['index_path'], key+'.index.jsonl')))
    result = dict(status='verified', job_id=job['job_id'], source_key=job['source_key'],
        source_sha256=job['source_sha256'], source_url=job['source_url'],
        encoder_identity=identity, outputs=outputs, counts=dict(counts),
        validation_split_scheme=job['validation_split_scheme'],
        language_model_sha256=plan['language_model_sha256'], language_minimum_probability=0.8,
        training_accepted=False)
    path = work/'receipt.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    receipt = publish(client, plan['bucket'], path, prefix+'/receipts/'+job['job_id']+'.json')
    client.put_object(Bucket=plan['bucket'], Key=prefix+'/progress/'+job['job_id']+'.json',
                      Body=json.dumps(result | {'receipt': receipt}).encode())
    # Raw source remains pinned and available at its source URL; durable native
    # outputs and receipt precede local cleanup.
    for p in work.iterdir():
        if p.is_file():
            p.unlink()
    return states[False]['tokens']


def main():
    import fasttext
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--work', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    for path_key, digest_key in [('language_model_path','language_model_sha256'),
                                ('tokenizer_path','tokenizer_sha256'), ('jobs_path','jobs_sha256')]:
        if file_sha(plan[path_key]) != plan[digest_key]:
            raise ValueError('textbook input identity changed: '+path_key)
    classifier = fasttext.load_model(plan['language_model_path'])
    identity = Native65kTokenizer(plan['tokenizer_path'], expected_sha256=plan['tokenizer_sha256']).identity()
    encoder = native_batch_pool(plan['tokenizer_path'], plan['tokenizer_sha256'], plan['encoding_threads'])
    client = boto3.client('s3', region_name='us-east-1')
    jobs = [json.loads(line) for line in Path(plan['jobs_path']).read_text().splitlines()]
    jobs.sort(key=lambda j: hashlib.sha256((plan['seed']+j['job_id']).encode()).hexdigest())
    total = 0
    for job in jobs:
        total += convert(job, plan, client, Path(args.work)/job['job_id'], classifier, encoder, identity)
        client.put_object(Bucket=plan['bucket'], Key=plan['output_prefix']+'/STATUS.json',
            Body=json.dumps(dict(train_tokens=total, target=plan['training_stock_target'],
                completed=total >= plan['training_stock_target'])).encode())
        if total >= plan['training_stock_target']:
            break
    if total < plan['training_stock_target']:
        raise RuntimeError('eligible English textbook supply below requested stock')


if __name__ == '__main__':
    main()
