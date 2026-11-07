"""Materialize an external curriculum from verified native conversion receipts.

This emits candidate shards. The final training manifest requires the separate
corpus acceptance audit and the native segment-aware loader acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

import boto3
from botocore.exceptions import ClientError
import numpy as np

from production.document_packing import ShardWriter, best_fit, file_sha, fragments
from production.native_corpus import publish, _output_matches


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def integer_split(total, weights):
    if total < 0 or not weights or any(x <= 0 for x in weights.values()):
        raise ValueError("invalid integer quota weights")
    denom = sum(weights.values())
    result = {k: total * v // denom for k, v in weights.items()}
    ranked = sorted(weights, key=lambda k: (-(total * weights[k] % denom), k))
    for k in ranked[:total - sum(result.values())]:
        result[k] += 1
    return result


def bounded_map(pool, function, items, ahead=8):
    iterator = iter(items)
    pending = deque()
    for _ in range(ahead):
        try:
            pending.append(pool.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        yield pending.popleft().result()
        try:
            pending.append(pool.submit(function, next(iterator)))
        except StopIteration:
            pass


class Exclusions:
    def __init__(self, path, expected_sha):
        if file_sha(path) != expected_sha:
            raise ValueError("frozen heldout exclusion identity changed")
        self.families = set()
        self.docs = set()
        self.rows = set()
        with Path(path).open() as source:
            for line in source:
                item = json.loads(line)
                self.families.add(item['family_id'])
                for location in item['locations']:
                    if location.get('doc_id'):
                        self.docs.add(str(location['doc_id']))
                    if location.get('row_key'):
                        self.rows.add(str(location['row_key']))

    def excludes(self, record):
        if record.get('validation') is not False:
            return True
        doc = str(record['doc_id'])
        row = record['row_key']
        family = str(record.get('split_family_id', doc))
        return (doc in self.docs or row in self.rows or family in self.docs
            or family in self.families
            or fingerprint(['published_id', doc]) in self.families
            or fingerprint(['explicit', family]) in self.families
            or fingerprint(['row', row]) in self.families)


class Worker:
    def __init__(self, plan, work, profile=None):
        self.plan = plan
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.client = boto3.Session(profile_name=profile, region_name='us-east-1').client('s3')
        self.bucket = plan['bucket']
        self.prefix = plan['output_prefix'].rstrip('/')
        self.exclusions = Exclusions(plan['exclusions_path'], plan['exclusions_sha256'])
        self.reservation_only = bool(plan.get('reservation_only', False))
        self.allocation = plan.get('allocation_id', 'initial_dense_data')
        ledger = self.work / 'selected.sqlite'
        if plan.get('input_ledger'):
            if ledger.exists():
                raise ValueError('existing work directory cannot replace the predecessor ledger')
            self.download(plan['input_ledger'], ledger)
        self.db = sqlite3.connect(ledger)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        cache_mib = plan.get('sqlite_cache_mib', 512)
        if type(cache_mib) is not int or not 32 <= cache_mib <= 16384:
            raise ValueError('invalid SQLite cache budget')
        self.db.execute('PRAGMA cache_size=-'+str(cache_mib*1024))
        self.db.execute('CREATE TABLE IF NOT EXISTS selected (text_sha TEXT PRIMARY KEY, row_key TEXT, leaf TEXT) WITHOUT ROWID')
        # Never continue a partial local cursor implicitly: restarting an
        # interrupted campaign needs an explicitly bound replay plan.
        if not plan.get('input_ledger') and self.db.execute('SELECT COUNT(*) FROM selected').fetchone()[0]:
            raise ValueError('nonempty selection ledger requires an explicit resume plan')
        if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('predecessor ledger is not a valid SQLite database')
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(selected)')}
        if 'allocation' not in columns:
            self.db.execute("ALTER TABLE selected ADD COLUMN allocation TEXT NOT NULL DEFAULT 'prior'")
        self.pending_suffixes = []
        if plan.get('input_suffixes'):
            path = self.work/'prior-suffixes.jsonl'
            self.download(plan['input_suffixes'], path)
            self.pending_suffixes = [json.loads(line) for line in path.read_text().splitlines()]
            seen = set()
            for row in self.pending_suffixes:
                sha = row['source_text_sha256']
                selected = self.db.execute('SELECT row_key,leaf,allocation FROM selected WHERE text_sha=?', (sha,)).fetchone()
                if (sha in seen or not selected or selected[0] != row['row_key']
                        or selected[1] != row['leaf'] or selected[2].startswith('protected:')
                        or not 0 < row['next_document_offset'] < row['original_content_tokens']
                        or row['remaining_content_tokens'] != row['original_content_tokens']-row['next_document_offset']):
                    raise ValueError('predecessor suffix does not match its allocation ledger')
                seen.add(sha)
        (self.work/'source-receipts').mkdir(exist_ok=True)
        self.start = time.monotonic()
        self.events = self.work / 'events.jsonl'
        self.receipts = []
        self.part_counter = 0
        self.totals = Counter()
        self.current = None
        self.last_heartbeat = 0

    def event(self, kind, **extra):
        row = {'event': kind, 'elapsed_seconds': time.monotonic()-self.start,
               'utc_unix': time.time(), **extra}
        with self.events.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)
        self.client.put_object(Bucket=self.bucket, Key=self.prefix+'/progress.json',
            Body=json.dumps({'status': kind, 'totals': dict(self.totals), **row}).encode())

    def deadline(self):
        if time.time() >= self.plan['deadline_unix'] - 180:
            raise TimeoutError('packing deadline: retaining durable candidate shards')

    def download(self, spec, path):
        bucket, key = spec['uri'][5:].split('/', 1)
        self.client.download_file(bucket, key, str(path))
        if path.stat().st_size != spec['bytes'] or file_sha(path) != spec['sha256']:
            raise ValueError('source payload/index hash mismatch: '+spec['uri'])

    def receipt(self, job):
        candidates = []
        for prefix in self.plan['conversion_prefixes']:
            key = prefix.rstrip('/')+'/progress/'+job['job_id']+'.json'
            try:
                raw = self.client.get_object(Bucket=self.bucket, Key=key)['Body'].read()
            except ClientError as exc:
                if exc.response['Error']['Code'] in ('NoSuchKey', '404'):
                    continue
                raise
            result = json.loads(raw)
            if result.get('status') != 'verified':
                continue
            if (result['source_key'] != job['source_key'] or result['job_id'] != job['job_id']
                or result['encoder_identity']['tokenizer_sha256'] != self.plan['tokenizer_sha256']
                or not result['encoder_identity'].get('roundtrip_checked_per_document')
                or not (result.get('original_holdout_assignments_preserved')
                        or (job.get('kind') == 'fresh_native_source'
                            and result.get('validation_split_scheme') == job.get('validation_split_scheme')
                            and result.get('source_sha256') == job['source_sha256']))):
                raise ValueError('conversion receipt identity mismatch')
            if job.get('kind') != 'fresh_native_source':
                original = {x['index_key']: x for x in job['parts']}
                for out in result['outputs']:
                    if out['original_index_key'] not in original:
                        raise ValueError('conversion output not bound to selected source job')
                    if out['validation'] != original[out['original_index_key']]['validation']:
                        raise ValueError('conversion split changed')
            result['_packing_receipt'] = {'uri': 's3://'+self.bucket+'/'+key,
                'sha256': hashlib.sha256(raw).hexdigest()}
            candidates.append(result)
        if not candidates:
            return None
        canonical = {(o.get('original_index_key', o['index']['uri']), o['bin']['sha256'], o['index']['sha256'])
                     for o in candidates[0]['outputs']}
        if any({(o.get('original_index_key', o['index']['uri']), o['bin']['sha256'], o['index']['sha256'])
                for o in r['outputs']} != canonical for r in candidates[1:]):
            raise ValueError('conflicting successful conversion receipts')
        return candidates[-1]

    def finish_shard(self):
        if self.current is None:
            return
        result = self.current.close()
        for key in ('payload', 'index'):
            spec = result[key]
            result[key] = publish(self.client, self.bucket, Path(spec['path']),
                self.prefix+'/parts/'+Path(spec['path']).name)
        result.update(leaf=self.leaf, stratum=self.stratum,
            tokenizer_sha256=self.plan['tokenizer_sha256'],
            exclusions_sha256=self.plan['exclusions_sha256'],
            plan_sha256=self.plan['plan_sha256'], source_order_seed=self.plan['seed'])
        path = self.work / ('receipt-'+str(self.part_counter)+'.json')
        path.write_text(json.dumps(result, indent=2)+'\n')
        spec = publish(self.client, self.bucket, path,
            self.prefix+'/receipts/'+str(self.part_counter).zfill(6)+'.json')
        self.receipts.append(result | {'receipt': spec})
        for key in ('sequences', 'content_tokens', 'valid_targets', 'used_slots', 'padding_slots'):
            self.totals[key] += result[key]
        self.current.payload.unlink()
        self.current.index.unlink()
        self.current = None
        self.event('shard_published', shard=self.part_counter, leaf=self.leaf,
                   stratum=self.stratum, content_tokens=result['content_tokens'], receipt=spec)

    def flush(self, buffer):
        if not buffer:
            return
        seed = fingerprint([self.plan['seed'], self.stratum, self.buffer_number])
        for pack in best_fit(buffer, length=self.plan['sequence_length'], seed=seed):
            if self.current is None:
                self.part_counter += 1
                self.current = ShardWriter(self.work/'shards',
                    'part-'+str(self.part_counter).zfill(6), length=self.plan['sequence_length'])
            self.current.add(pack)
            if self.current.counts['sequences'] >= self.plan['sequences_per_shard']:
                self.finish_shard()
        buffer.clear()
        self.buffer_number += 1
        self.deadline()

    def consume_suffixes(self, leaf, stratum, target, buffer):
        if self.reservation_only:
            return 0
        selected = 0
        retained = []
        for record in self.pending_suffixes:
            if record['leaf'] != leaf or record['stratum'] != stratum or selected == target:
                retained.append(record)
                continue
            spec = {'uri': record['source_bin_uri'], 'sha256': record['source_bin_sha256']}
            b, k = spec['uri'][5:].split('/', 1)
            spec['bytes'] = self.client.head_object(Bucket=b, Key=k)['ContentLength']
            path = self.work/'suffix.uint16'
            self.download(spec, path)
            tokens = np.memmap(path, dtype='<u2', mode='r')
            offset = record['next_document_offset']
            take = min(record['remaining_content_tokens'], target-selected)
            begin = record['source_token_start']+offset
            if begin+take > len(tokens):
                raise ValueError('suffix exceeds source payload')
            document = {k:v for k,v in record.items() if k not in (
                'next_document_offset','remaining_content_tokens','leaf','stratum')}
            for part in fragments(tokens[begin:begin+take], document,
                    length=self.plan['sequence_length'], offset=offset,
                    document_end=offset+take==record['original_content_tokens']):
                buffer.append(part)
                if len(buffer) >= self.plan['packing_buffer_fragments']:
                    self.flush(buffer)
            selected += take
            if take < record['remaining_content_tokens']:
                retained.append(record | {'next_document_offset':offset+take,
                    'remaining_content_tokens':record['remaining_content_tokens']-take})
            del tokens
            path.unlink()
        self.pending_suffixes = retained
        return selected

    def process_stratum(self, leaf, stratum, jobs, target):
        self.leaf, self.stratum = leaf, stratum
        self.buffer_number = 0
        buffer = []
        selected = self.consume_suffixes(leaf, stratum, target, buffer)
        rejected = Counter()
        ordered = [] if selected == target else sorted(jobs, key=lambda j: fingerprint([self.plan['seed'], j['job_id']]))
        self.event('stratum_started', leaf=leaf, stratum=stratum, target=target)
        # Prefetch metadata only. Raw payloads remain bounded to one part.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for job, result in zip(ordered, bounded_map(pool, self.receipt, ordered)):
                self.deadline()
                if result is None:
                    rejected['unverified_jobs'] += 1
                    continue
                provenance = self.work/'source-receipts'/job['job_id']
                provenance.parent.mkdir(exist_ok=True)
                provenance.write_text(json.dumps(result, separators=(',', ':')))
                publish(self.client, self.bucket, provenance,
                    self.prefix+'/source-receipts/'+job['job_id']+'.json')
                outputs = sorted((o for o in result['outputs'] if not o['validation']),
                    key=lambda o: fingerprint([self.plan['seed'], o.get('original_index_key', o['index']['uri'])]))
                for output in outputs:
                    index_path = self.work/'input.index.jsonl'
                    bin_path = self.work/'input.uint16'
                    self.download(output['index'], index_path)
                    rows = []
                    cursor = 0
                    document_count = 0
                    with index_path.open() as stream:
                        for line in stream:
                            row = json.loads(line)
                            if (type(row['n_tokens']) is not int or row['n_tokens'] <= 0
                                    or row['token_start'] != cursor or row['validation'] is not False):
                                raise ValueError('noncontiguous or invalid training index')
                            cursor += row['n_tokens']
                            document_count += 1
                            if self.exclusions.excludes(row):
                                rejected['heldout_documents'] += 1
                            elif self.db.execute('SELECT 1 FROM selected WHERE text_sha=?',
                                                 (row['source_text_sha256'],)).fetchone():
                                rejected['previously_allocated_documents'] += 1
                            else:
                                rows.append(row)
                    if (cursor != output['tokens'] or cursor*2 != output['bin']['bytes']
                            or document_count != output['documents']):
                        raise ValueError('index coverage differs from token payload')
                    if not rows:
                        continue
                    if self.reservation_only:
                        b,k = output['bin']['uri'][5:].split('/',1)
                        if not _output_matches(self.client.head_object(Bucket=b,Key=k,ChecksumMode='ENABLED'), output['bin']):
                            raise ValueError('reservation token payload identity changed')
                    else:
                        self.download(output['bin'], bin_path)
                        tokens = np.memmap(bin_path, dtype='<u2', mode='r')
                    rows.sort(key=lambda r: fingerprint([self.plan['seed'], r['row_key']]))
                    for row in rows:
                        if selected >= target:
                            break
                        text_sha = row['source_text_sha256']
                        try:
                            self.db.execute('INSERT INTO selected(text_sha,row_key,leaf,allocation) VALUES(?,?,?,?)',
                                (text_sha, row['row_key'], leaf,
                                 ('protected:' if self.reservation_only else 'allocated:')+self.allocation))
                        except sqlite3.IntegrityError:
                            rejected['exact_duplicate_documents'] += 1
                            continue
                        take = min(row['n_tokens'], target-selected)
                        begin = row['token_start']
                        document = {k: row[k] for k in ('doc_id', 'row_key',
                            'split_family_id', 'source_text_sha256')}
                        document.update(source_bin_uri=output['bin']['uri'],
                            source_bin_sha256=output['bin']['sha256'],
                            source_index_sha256=output['index']['sha256'],
                            source_token_start=begin, original_content_tokens=row['n_tokens'])
                        if self.reservation_only:
                            reservation = document | {'leaf':leaf, 'stratum':stratum,
                                'allocation':self.allocation, 'allocated_content_tokens':take,
                                'whole_record_protected':True}
                            with (self.work/'protected-records.jsonl').open('a') as stream:
                                stream.write(json.dumps(reservation,separators=(',',':'))+'\n')
                            self.totals['reserved_content_tokens'] += take
                            self.totals['protected_whole_record_tokens'] += row['n_tokens']
                            selected += take
                            continue
                        for piece in fragments(tokens[begin:begin+take], document,
                                length=self.plan['sequence_length'], document_end=take==row['n_tokens']):
                            buffer.append(piece)
                            if len(buffer) >= self.plan['packing_buffer_fragments']:
                                self.flush(buffer)
                        selected += take
                        if take < row['n_tokens']:
                            suffix = document | {'next_document_offset': take,
                                'remaining_content_tokens': row['n_tokens']-take,
                                'leaf': leaf, 'stratum': stratum}
                            self.pending_suffixes.append(suffix)
                    self.db.commit()
                    if not self.reservation_only:
                        del tokens
                        bin_path.unlink()
                    index_path.unlink()
                    if time.monotonic()-self.last_heartbeat > 30:
                        self.event('packing', leaf=leaf, stratum=stratum,
                            selected_content_tokens=selected, target=target,
                            rejections=dict(rejected))
                        self.last_heartbeat = time.monotonic()
                    if selected >= target:
                        break
                if selected >= target:
                    break
        self.flush(buffer)
        self.finish_shard()
        result = {'leaf': leaf, 'stratum': stratum, 'target': target, 'selected': selected,
                  'shortfall': target-selected, 'rejections': dict(rejected)}
        self.event('stratum_finished', **result)
        return result

    def select_sources(self):
        groups = defaultdict(list)
        with Path(self.plan['jobs_path']).open() as stream:
            for line in stream:
                job = json.loads(line)
                groups[(job['packing_leaf'], job['packing_stratum'])].append(job)
        all_results = []
        targets = dict(self.plan['source_targets'])
        fallbacks = self.plan.get('source_fallbacks', {})
        if any(a not in targets or b not in targets or b <= a for a,b in fallbacks.items()):
            raise ValueError('fallback must name a later assigned source in the same phase')
        # Fixed order makes global exact-text duplicate ownership reproducible.
        for leaf in sorted(targets):
            target = targets[leaf]
            first_result = len(all_results)
            selected_groups = {s: js for (l, s), js in groups.items() if l == leaf}
            if not selected_groups:
                all_results.append({'leaf': leaf, 'target': target, 'selected': 0,
                    'shortfall': target, 'reason': 'source_conversion_pending'})
                self.event('source_pending', leaf=leaf, target=target)
            else:
                weights = {s: sum(sum(p['token_snapshot']['bytes']//2
                                 for p in j.get('parts', []) if not p['validation'])
                                 if j.get('kind') != 'fresh_native_source' else 1 for j in js)
                           for s, js in selected_groups.items()}
                quotas = integer_split(target, weights)
                for stratum, quota in sorted(quotas.items()):
                    if quota:
                        all_results.append(self.process_stratum(leaf, stratum,
                                             selected_groups[stratum], quota))
            deficit = sum(r['shortfall'] for r in all_results[first_result:])
            if deficit and leaf in fallbacks:
                replacement = fallbacks[leaf]
                targets[replacement] += deficit
                for result in all_results[first_result:]:
                    result.update(original_target=result['target'],
                        reallocated_tokens=result['shortfall'], reallocated_to=replacement,
                        target=result['selected'], shortfall=0)
                self.event('authorized_source_fallback', source=leaf,
                           replacement=replacement, content_tokens=deficit)
        return all_results

    def materialize_fixed(self):
        spec = self.plan['fixed_selection']
        path = self.work/'fixed-selection.jsonl'
        self.download(spec, path)
        allocation = self.plan['fixed_allocation_id']
        current_bin = None
        tokens = None
        buffer = []
        counts = Counter()
        self.buffer_number = 0
        self.leaf = self.stratum = None
        try:
            with path.open() as stream:
                for line in stream:
                    self.deadline()
                    record = json.loads(line)
                    if record['allocation'] != allocation or not record['whole_record_protected']:
                        raise ValueError('fixed selection has a different protected allocation')
                    row = self.db.execute('SELECT leaf,allocation FROM selected WHERE text_sha=?',
                        (record['source_text_sha256'],)).fetchone()
                    if row != (record['leaf'], 'protected:'+allocation):
                        raise ValueError('fixed selection is not protected in its predecessor ledger')
                    if self.exclusions.excludes(record | {'validation':False}):
                        raise ValueError('protected record overlaps frozen heldouts')
                    if (record['leaf'],record['stratum']) != (self.leaf,self.stratum):
                        self.flush(buffer)
                        self.finish_shard()
                        self.leaf,self.stratum = record['leaf'],record['stratum']
                        self.buffer_number = 0
                    if record['source_bin_uri'] != current_bin:
                        if tokens is not None:
                            del tokens
                            tokens = None
                        bin_path = self.work/'fixed.uint16'
                        b,k = record['source_bin_uri'][5:].split('/',1)
                        source = {'uri':record['source_bin_uri'],
                            'sha256':record['source_bin_sha256'],
                            'bytes':self.client.head_object(Bucket=b,Key=k)['ContentLength']}
                        self.download(source, bin_path)
                        tokens = np.memmap(bin_path,dtype='<u2',mode='r')
                        current_bin = record['source_bin_uri']
                    n = record['allocated_content_tokens']
                    start = record['source_token_start']
                    if not 0 < n <= record['original_content_tokens'] or start+n > len(tokens):
                        raise ValueError('invalid fixed source span')
                    document = {k:v for k,v in record.items() if k not in
                        ('allocation','allocated_content_tokens','whole_record_protected','leaf','stratum')}
                    for piece in fragments(tokens[start:start+n],document,
                            length=self.plan['sequence_length'],document_end=n==record['original_content_tokens']):
                        buffer.append(piece)
                        if len(buffer) >= self.plan['packing_buffer_fragments']:
                            self.flush(buffer)
                    counts[record['leaf']] += n
                    if counts[record['leaf']] > self.plan['source_targets'][record['leaf']]:
                        raise ValueError('fixed selection exceeds declared source quota')
            self.flush(buffer)
            self.finish_shard()
        finally:
            if tokens is not None:
                del tokens
        return [{'leaf':leaf,'target':target,'selected':counts[leaf],
                 'shortfall':target-counts[leaf]} for leaf,target in self.plan['source_targets'].items()]

    def run(self):
        all_results = self.materialize_fixed() if self.plan.get('fixed_selection') else self.select_sources()
        self.db.commit()
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()
        suffix_path = self.work/'quota-suffix-reservations.jsonl'
        suffix_path.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n'
                                      for x in self.pending_suffixes))
        artifacts = {}
        for path in [self.events, self.work/'selected.sqlite',
                     suffix_path, self.work/'protected-records.jsonl']:
            if path.exists():
                artifacts[path.name] = publish(self.client, self.bucket, path, self.prefix+'/audit/'+path.name)
        archive = shutil.make_archive(str(self.work/'source-receipts'), 'gztar',
                                      self.work/'source-receipts')
        publish(self.client, self.bucket, Path(archive), self.prefix+'/audit/source-receipts.tar.gz')
        report = {'schema': 'terminus_warmup_materialization_v1',
            'plan_sha256': self.plan['plan_sha256'], 'results': all_results,
            'totals': dict(self.totals), 'shards': self.receipts,
            'physical_creation_finished': not self.reservation_only,
            'reservation_creation_finished': self.reservation_only,
            'allocation_id':self.allocation, 'artifacts':artifacts,
            'input_ledger':self.plan.get('input_ledger'),
            'all_curriculum_quotas_satisfied': all(x['shortfall'] == 0 for x in all_results),
            'cross_document_mask_required': True, 'training_manifest_published': False,
            'native_training_accepted': False}
        path = self.work/'MATERIALIZATION.json'
        path.write_text(json.dumps(report, indent=2)+'\n')
        publish(self.client, self.bucket, path, self.prefix+'/MATERIALIZATION.json')
        self.event('finished', quotas_satisfied=report['all_curriculum_quotas_satisfied'])
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--profile')
    args = parser.parse_args()
    path = Path(args.plan)
    plan = json.loads(path.read_text())
    plan['plan_sha256'] = file_sha(path)
    for key, digest_key in [('jobs_path', 'jobs_sha256'),
                            ('exclusions_path', 'exclusions_sha256')]:
        if file_sha(plan[key]) != plan[digest_key]:
            raise ValueError('plan input hash mismatch: '+key)
    worker = Worker(plan, args.work, args.profile)
    try:
        worker.run()
    except BaseException as exc:
        worker.event('failed', error=type(exc).__name__+': '+str(exc))
        raise


if __name__ == '__main__':
    main()
