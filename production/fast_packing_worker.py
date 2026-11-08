"""RAM-ledger, bounded prefetch and multiprocess document packing.

The reference worker remains available for equivalence checks. This worker
keeps the same selection/packing order and emits the same segment contract.
Only closed, verified snapshots and fully uploaded shards become dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import gzip
import json
import multiprocessing
from pathlib import Path
import shutil
import threading
import time
import uuid

import boto3
from botocore.config import Config
import numpy as np

from production.compact_ledger import CompactLedger, key, row_key_hash
from production.readonly_ledger import ReadOnlyLedger
from production.parallel_gzip import ParallelGzipWriter
from production.document_packing import ShardWriter, file_sha, fragments
from production.native_corpus import publish, _output_matches
from production.packing_parallel import (download_verified, initialize,
    pack_chunk, prepare_index, prepare_records, FragmentReference)
from production.packing_worker import Worker as ReferenceWorker, Exclusions, fingerprint, bounded_map
from production.preparation_queue import object_spec


class FastWorker(ReferenceWorker):
    def __init__(self, plan, work, profile=None):
        self.plan=plan;self.work=Path(work);self.work.mkdir(parents=True,exist_ok=True)
        if any(self.work.iterdir()):raise ValueError('nonempty working directory requires an explicit recovery plan')
        self.client=boto3.Session(profile_name=profile,region_name='us-east-1').client(
            's3',config=Config(max_pool_connections=32,connect_timeout=10,read_timeout=60,
                retries={'mode':'standard','max_attempts':5}))
        self.bucket=plan['bucket'];self.prefix=plan['output_prefix'].rstrip('/')
        self.exclusions=Exclusions(plan['exclusions_path'],plan['exclusions_sha256'])
        self.reservation_only=bool(plan.get('reservation_only',False))
        self.allocation=plan.get('allocation_id','initial_dense_data')
        self.start=time.monotonic();self.events=self.work/'events.jsonl'
        self.scratch=Path(plan.get('scratch_directory','/dev/shm'))/('ci1-pack-'+uuid.uuid4().hex)
        self.scratch.mkdir(parents=True)
        self.ram_limit=plan.get('input_ram_mib',8192)<<20;self.ram_used=0
        self.paths={};self.path_lock=threading.Lock()
        self.path_holds=Counter();self.pending_release=set();self.buffer_paths=set()
        predecessor=None
        if plan.get('input_ledger'):
            predecessor=self.work/'input-ledger.sqlite'
            download_verified(self.client,plan['input_ledger'],predecessor)
        self.readonly_ledger=bool(plan.get('read_only_fixed_ledger',False))
        if self.readonly_ledger:
            if not plan.get('fixed_selection') or self.reservation_only or predecessor is None:
                raise ValueError('immutable ledger is only valid for fixed materialization')
            self.ledger=ReadOnlyLedger(predecessor,
                cache_mib=plan.get('readonly_ledger_cache_mib',512),
                mmap_mib=plan.get('readonly_ledger_mmap_mib',2048))
        else:
            self.ledger=CompactLedger(self.work,predecessor,
                memory_mib=plan.get('ledger_memory_mib',65536),
                seen_mib=plan.get('ledger_seen_mib',8192),
                cache_mib=plan.get('ledger_fallback_cache_mib',24576),
                use_seen=not bool(plan.get('fixed_selection')),
                commit_rows=plan.get('ledger_commit_rows',0),
                bulk_insert_rows=plan.get('ledger_bulk_insert_rows',0))
        self.pending_suffixes=[]
        if plan.get('input_suffixes'):
            path=self.work/'prior-suffixes.jsonl';self.download(plan['input_suffixes'],path)
            self.pending_suffixes=[json.loads(line)for line in path.read_text().splitlines()]
            seen=set()
            for row in self.pending_suffixes:
                sha=row['source_text_sha256'];entry=self.ledger.get(sha)
                if (sha in seen or not entry or entry[0]!=row_key_hash(row['row_key'])
                    or entry[1]!=row['leaf']
                    or (entry[2].startswith('protected:') and entry[2]!='protected:'+row.get('protected_allocation',''))
                    or (row.get('protected_allocation') and entry[2]!='protected:'+row['protected_allocation'])
                    or not 0<row['next_document_offset']<row['original_content_tokens']
                    or row['remaining_content_tokens']!=row['original_content_tokens']-row['next_document_offset']):
                    raise ValueError('suffix differs from predecessor allocation')
                seen.add(sha)
        context=multiprocessing.get_context('spawn')
        self.metadata_pool=ProcessPoolExecutor(max_workers=plan.get('metadata_workers',4),
            mp_context=context,initializer=initialize,
            initargs=(plan['exclusions_path'],plan['exclusions_sha256']))
        self.packing_pool=ProcessPoolExecutor(max_workers=plan.get('packing_workers',8),mp_context=context)
        self.io_pool=ThreadPoolExecutor(max_workers=plan.get('io_workers',6))
        self.upload_pool=ThreadPoolExecutor(max_workers=plan.get('upload_workers',3))
        self.pack_pending=deque();self.upload_pending=deque()
        self.receipts=[];self.part_counter=0;self.totals=Counter();self.current=None
        self.last_heartbeat=0;self.source_receipt_ids=set()
        self.source_receipts_dir=self.work/'source-receipts';self.source_receipts_dir.mkdir(exist_ok=True)
        self.provenance=(self.source_receipts_dir/'verified.jsonl').open('wb',buffering=64<<20)
        self.reserved_path=self.work/('protected-records.jsonl.gz' if plan.get('compress_reservations') else 'protected-records.jsonl')
        self.reserved=((ParallelGzipWriter(self.reserved_path,workers=plan.get('compression_workers',4))
                       if plan.get('parallel_reservation_compression')else gzip.open(self.reserved_path,'wb',compresslevel=1)) if plan.get('compress_reservations')
                       else self.reserved_path.open('wb',buffering=64<<20)) if self.reservation_only else None

    def download(self, spec, path):
        return download_verified(self.client,spec,path)

    def input_path(self, spec, suffix):
        with self.path_lock:
            size=spec['bytes'];ram=self.ram_used+size<=self.ram_limit
            root=self.scratch if ram else self.work/'input-spill'
            root.mkdir(parents=True,exist_ok=True)
            path=root/(uuid.uuid4().hex+suffix)
            self.paths[str(path)]=size if ram else 0
            if ram:self.ram_used+=size
        return path

    def release(self, path):
        path=Path(path)
        with self.path_lock:
            if self.path_holds[str(path)]:
                self.pending_release.add(str(path));return
            path.unlink(missing_ok=True)
            self.ram_used-=self.paths.pop(str(path),0)
            self.pending_release.discard(str(path));self.path_holds.pop(str(path),None)

    def retain_buffer_input(self,path):
        path=str(path)
        if path not in self.buffer_paths:
            self.buffer_paths.add(path)
            with self.path_lock:self.path_holds[path]+=1

    def release_chunk_inputs(self,paths):
        for path in paths:
            with self.path_lock:
                self.path_holds[path]-=1
                if self.path_holds[path]<0:raise ValueError('invalid packing input lifetime')
                remove=not self.path_holds[path]and path in self.pending_release
            if remove:self.release(path)

    def provenance_receipt(self, job, result):
        if job['job_id']not in self.source_receipt_ids:
            self.provenance.write((json.dumps(result,separators=(',',':'))+'\n').encode())
            self.source_receipt_ids.add(job['job_id'])

    def raw_outputs(self, jobs):
        with ThreadPoolExecutor(max_workers=8)as pool:
            for job,result in zip(jobs,bounded_map(pool,self.receipt,jobs)):
                self.deadline()
                if result is None:
                    yield None,None;continue
                self.provenance_receipt(job,result)
                for output in sorted((o for o in result['outputs']if not o['validation']),
                    key=lambda o:fingerprint([self.plan['seed'],o.get('original_index_key',o['index']['uri'])])):
                    yield output,result

    def prepared_outputs(self, jobs, rejected):
        pending=deque();pending_bytes=0
        def submit(output):
            path=self.input_path(output['index'],'.index.jsonl')
            def receive():
                self.download(output['index'],path)
                return self.metadata_pool.submit(prepare_index,str(path),self.plan['seed'])
            return output,path,self.io_pool.submit(receive)
        def consume(item):
            output,path,future=item
            try:prepared=future.result().result()
            finally:self.release(path)
            if (prepared['tokens']!=output['tokens'] or prepared['tokens']*2!=output['bin']['bytes']
                    or prepared['documents']!=output['documents']):
                raise ValueError('prepared index coverage differs')
            rejected['heldout_documents']+=prepared['heldout']
            return output,prepared['rows']
        try:
            for output,_ in self.raw_outputs(jobs):
                if output is None:rejected['unverified_jobs']+=1;continue
                if self.ledger.drained(output['index']['sha256'],output['bin']['sha256']):
                    rejected['previously_drained_outputs']+=1;continue
                estimate=8*output['index']['bytes']
                while pending and (pending_bytes+estimate>self.plan.get('metadata_ram_mib',12288)<<20):
                    item=pending.popleft();pending_bytes-=item[0]['index']['bytes']*8
                    yield consume(item)
                pending.append(submit(output))
                pending_bytes+=estimate
                if len(pending)>=self.plan.get('metadata_prefetch',2):
                    item=pending.popleft();pending_bytes-=item[0]['index']['bytes']*8
                    yield consume(item)
            while pending:yield consume(pending.popleft())
        finally:
            for _,path,future in pending:
                try:future.result().result()
                finally:self.release(path)

    def selected_outputs(self, jobs, leaf, stratum, target, initial, rejected):
        allocated=initial
        for output,rows in self.prepared_outputs(jobs,rejected):
            known=self.ledger.contains_many([r['_sha']for r in rows]);chosen=[];local=set();exhausted=True
            for row in rows:
                if allocated>=target:exhausted=False;break
                sha=row['_sha']
                if sha in known:rejected['previously_allocated_documents']+=1;continue
                if sha in local:rejected['exact_duplicate_documents']+=1;continue
                local.add(sha);take=min(row['n_tokens'],target-allocated)
                chosen.append((row,take));allocated+=take
            if chosen:
                self.ledger.add_many([r for r,_ in chosen],leaf,
                    ('protected:'if self.reservation_only else 'allocated:')+self.allocation)
            if exhausted:self.ledger.mark_drained(output['index']['sha256'],output['bin']['sha256'])
            if not chosen:continue
            if self.reservation_only:
                def verify(spec=output['bin']):
                    b,k=spec['uri'][5:].split('/',1)
                    if not _output_matches(self.client.head_object(Bucket=b,Key=k,ChecksumMode='ENABLED'),spec):
                        raise ValueError('reservation token payload differs')
                future=self.io_pool.submit(verify);path=None
            else:
                path=self.input_path(output['bin'],'.uint16')
                future=self.io_pool.submit(self.download,output['bin'],path)
            yield output,chosen,path,future
            if allocated>=target:break

    def process_stratum(self, leaf, stratum, jobs, target):
        self.leaf,self.stratum=leaf,stratum;self.buffer_number=0;buffer=[]
        selected=self.consume_suffixes(leaf,stratum,target,buffer);rejected=Counter()
        ordered=[]if selected==target else sorted(jobs,key=lambda j:fingerprint([self.plan['seed'],j['job_id']]))
        self.event('stratum_started',leaf=leaf,stratum=stratum,target=target)
        source=self.selected_outputs(ordered,leaf,stratum,target,selected,rejected);pending=deque()
        def consume(item):
            nonlocal selected
            output,chosen,path,future=item;future.result()
            tokens=np.memmap(path,dtype='<u2',mode='r')if path else None
            try:
                for row,take in chosen:
                    document={k:row[k]for k in ('doc_id','row_key','split_family_id','source_text_sha256')}
                    document.update(source_bin_uri=output['bin']['uri'],source_bin_sha256=output['bin']['sha256'],
                        source_index_sha256=output['index']['sha256'],source_token_start=row['token_start'],
                        original_content_tokens=row['n_tokens'])
                    if self.reservation_only:self.reserve(document,leaf,stratum,take,0)
                    else:
                        for part in fragments(tokens[row['token_start']:row['token_start']+take],document,
                            length=self.plan['sequence_length'],document_end=take==row['n_tokens']):
                            buffer.append(part)
                            if len(buffer)>=self.plan['packing_buffer_fragments']:self.flush(buffer)
                    selected+=take
                    if take<row['n_tokens']and(not self.reservation_only or self.plan.get('reservation_reuse_suffixes')):
                        suffix=document|{'next_document_offset':take,'remaining_content_tokens':row['n_tokens']-take,
                            'leaf':leaf,'stratum':stratum}
                        if self.reservation_only:suffix['protected_allocation']=self.allocation
                        self.pending_suffixes.append(suffix)
            finally:
                if tokens is not None:del tokens
                if path:self.release(path)
            if time.monotonic()-self.last_heartbeat>30:
                self.event('packing',leaf=leaf,stratum=stratum,selected_content_tokens=selected,target=target,
                    rejections=dict(rejected));self.last_heartbeat=time.monotonic()
        try:
            for item in source:
                pending.append(item)
                if len(pending)>=self.plan.get('payload_prefetch',2):consume(pending.popleft())
            while pending:consume(pending.popleft())
        finally:
            source.close()
        self.flush(buffer);self.finish_stratum()
        result={'leaf':leaf,'stratum':stratum,'target':target,'selected':selected,
            'shortfall':target-selected,'rejections':dict(rejected)}
        self.event('stratum_finished',**result);return result

    def reserve(self, document, leaf, stratum, take, offset):
        import orjson
        reservation=document|{'leaf':leaf,'stratum':stratum,'allocation':self.allocation,
            'allocated_content_tokens':take,'whole_record_protected':True}
        if offset:reservation['allocated_document_offset']=offset
        self.reserved.write(orjson.dumps(reservation,option=orjson.OPT_APPEND_NEWLINE))
        self.totals['reserved_content_tokens']+=take
        self.totals['protected_whole_record_tokens']+=document['original_content_tokens']

    def consume_suffixes(self, leaf, stratum, target, buffer):
        if self.reservation_only and not self.plan.get('reservation_reuse_suffixes'):return 0
        selected=0;retained=[]
        for record in self.pending_suffixes:
            if record['leaf']!=leaf or record['stratum']!=stratum or selected==target:
                retained.append(record);continue
            spec={'uri':record['source_bin_uri'],'sha256':record['source_bin_sha256']}
            b,k=spec['uri'][5:].split('/',1);spec['bytes']=self.client.head_object(Bucket=b,Key=k)['ContentLength']
            offset=record['next_document_offset'];take=min(record['remaining_content_tokens'],target-selected)
            begin=record['source_token_start']+offset
            if begin+take>spec['bytes']//2:raise ValueError('invalid suffix span')
            document={k:v for k,v in record.items()if k not in
                ('next_document_offset','remaining_content_tokens','leaf','stratum','protected_allocation')}
            if self.reservation_only:
                if object_spec(self.client,b,k)['sha256']!=spec['sha256']:raise ValueError('suffix payload differs')
                self.ledger.change_owner(record['source_text_sha256'],'protected:'+self.allocation)
                self.reserve(document,leaf,stratum,take,offset)
            else:
                path=self.input_path(spec,'.suffix.uint16');self.download(spec,path)
                tokens=np.memmap(path,dtype='<u2',mode='r')
                for part in fragments(tokens[begin:begin+take],document,length=self.plan['sequence_length'],
                    offset=offset,document_end=offset+take==record['original_content_tokens']):
                    buffer.append(part)
                    if len(buffer)>=self.plan['packing_buffer_fragments']:self.flush(buffer)
                del tokens;self.release(path)
            selected+=take
            if take<record['remaining_content_tokens']:
                remainder=record|{'next_document_offset':offset+take,
                    'remaining_content_tokens':record['remaining_content_tokens']-take}
                if self.reservation_only:remainder['protected_allocation']=self.allocation
                retained.append(remainder)
        self.pending_suffixes=retained;return selected

    def flush(self, buffer):
        if not buffer:return
        seed=fingerprint([self.plan['seed'],self.stratum,self.buffer_number])
        future=self.packing_pool.submit(pack_chunk,tuple(buffer),self.plan['sequence_length'],seed,str(self.scratch))
        self.pack_pending.append((future,self.leaf,self.stratum,self.buffer_paths))
        self.buffer_paths=set();buffer.clear();self.buffer_number+=1
        if len(self.pack_pending)>=self.plan.get('packing_prefetch',6):self.drain_pack()
        self.deadline()

    def drain_pack(self):
        future,leaf,stratum,paths=self.pack_pending.popleft()
        try:chunk=future.result()
        finally:self.release_chunk_inputs(paths)
        start=0
        if self.current is not None and (leaf,stratum)!=(self.current_leaf,self.current_stratum):
            self.finish_shard()
        data_hash=hashlib.sha256();meta_hash=hashlib.sha256()
        with Path(chunk['payload']['path']).open('rb')as data,Path(chunk['index']['path']).open('rb')as meta:
            while start<chunk['sequences']:
                if self.current is None:
                    self.part_counter+=1
                    self.current=ShardWriter(self.scratch,'part-'+str(self.part_counter).zfill(6),
                        length=self.plan['sequence_length'],buffer_bytes=64<<20,write_offsets=True)
                    self.current_leaf,self.current_stratum=leaf,stratum
                n=min(chunk['sequences']-start,self.plan['sequences_per_shard']-self.current.counts['sequences'])
                stop=start+n;byte_count=n*self.plan['sequence_length']*2
                payload=data.read(byte_count)
                meta_count=int(chunk['offsets'][stop]-chunk['offsets'][start])
                metadata=meta.read(meta_count)
                if len(payload)!=byte_count or len(metadata)!=meta_count:raise ValueError('truncated packing chunk')
                data_hash.update(payload);meta_hash.update(metadata)
                delta=chunk['prefixes'][stop]-chunk['prefixes'][start]
                counts={'sequences':n}|{name:int(value)for name,value in zip(chunk['names'],delta)}
                self.current.append_encoded(payload,metadata,counts,
                    offsets=chunk['offsets'][start+1:stop+1]-chunk['offsets'][start],
                    source_counts={key:int(values[stop]-values[start])for key,values in chunk.get('source_prefixes',{}).items()
                                   if values[stop]>values[start]});start=stop
                if self.current.counts['sequences']==self.plan['sequences_per_shard']:self.finish_shard()
        if data_hash.hexdigest()!=chunk['payload']['sha256']or meta_hash.hexdigest()!=chunk['index']['sha256']:
            raise ValueError('packing chunk differs from worker checksum')
        Path(chunk['payload']['path']).unlink();Path(chunk['index']['path']).unlink()

    def finish_shard(self):
        if self.current is None:return
        result=self.current.close();self.current=None;ordinal=self.part_counter
        leaf,stratum=self.current_leaf,self.current_stratum
        def upload():
            for name in ('payload','index','index_offsets'):
                spec=result[name];path=Path(spec['path'])
                result[name]=publish(self.client,self.bucket,path,self.prefix+'/parts/'+path.name,
                    expected_sha256=spec['sha256']);path.unlink()
            result.update(leaf=leaf,stratum=stratum,tokenizer_sha256=self.plan['tokenizer_sha256'],
                exclusions_sha256=self.plan['exclusions_sha256'],plan_sha256=self.plan['plan_sha256'],
                source_order_seed=self.plan['seed'])
            path=self.work/('receipt-'+str(ordinal)+'.json');path.write_text(json.dumps(result,indent=2)+'\n')
            spec=publish(self.client,self.bucket,path,self.prefix+'/receipts/'+str(ordinal).zfill(6)+'.json')
            return result|{'receipt':spec}
        self.upload_pending.append(self.upload_pool.submit(upload))
        if len(self.upload_pending)>=self.plan.get('upload_prefetch',3):self.drain_upload()

    def drain_upload(self):
        result=self.upload_pending.popleft().result();self.receipts.append(result)
        for name in ('sequences','content_tokens','valid_targets','used_slots','padding_slots','synthetic_tokens'):
            if result.get(name):self.totals[name]+=result[name]
        self.event('shard_published',shard=len(self.receipts),leaf=result['leaf'],stratum=result['stratum'],
            content_tokens=result['content_tokens'],receipt=result['receipt'])

    def finish_stratum(self,force=False):
        if self.plan.get('overlap_fixed_strata')and self.plan.get('fixed_selection')and not force:return
        while self.pack_pending:self.drain_pack()
        self.finish_shard()
        while self.upload_pending:self.drain_upload()

    def prepared_record_batches(self, path):
        pending=deque()
        opener=gzip.open(path,'rb') if str(path).endswith('.gz') else Path(path).open('rb',buffering=8<<20)
        with opener as stream:
            while True:
                lines=[];size=0
                while len(lines)<100000 and size<64<<20:
                    line=stream.readline()
                    if not line:break
                    lines.append(line);size+=len(line)
                if not lines:break
                pending.append(self.metadata_pool.submit(prepare_records,b''.join(lines)))
                if len(pending)>=2:yield pending.popleft().result()
        while pending:yield pending.popleft().result()

    def materialize_fixed(self):
        spec=self.plan['fixed_selection']
        if spec.get('compression') not in (None,'gzip'):raise ValueError('unsupported reservation compression')
        path=self.work/('fixed-selection.jsonl.gz' if spec.get('compression')=='gzip' else 'fixed-selection.jsonl')
        self.download(spec,path)
        owner='protected:'+self.plan['fixed_allocation_id'];counts=Counter();observed=Counter();buffer=[]
        paired=None
        if self.plan.get('translation_pairs'):
            from production.fixed_translation_pairs import FixedPairPreparation
            paired=FixedPairPreparation(self,path,self.plan['translation_pairs'])
            self.leaf='paired/translations';self.stratum='verified';self.buffer_number=0
            self.event('translation_pair_preparation_started',phase_ceiling_content_tokens=paired.policy['phase_ceiling_content_tokens'])
            for pair in paired.pairs():
                buffer.append(pair)
                if len(buffer)>=self.plan['packing_buffer_fragments']:self.flush(buffer)
                if time.monotonic()-self.last_heartbeat>30:
                    self.event('translation_pairs',pair_count=paired.pair_count,paired_content_tokens=sum(paired.charges.values()))
                    self.last_heartbeat=time.monotonic()
            self.flush(buffer);self.finish_stratum();counts.update(paired.charges)
            self.pair_report=paired.report()
            (self.work/'TRANSLATION_PAIRS.json').write_text(json.dumps(self.pair_report,indent=2)+'\n')
            self.event('translation_pair_preparation_finished',**self.pair_report)
        ranges=self.plan.get('fixed_source_ranges')
        if ranges is not None:
            if set(ranges)!=set(self.plan['source_targets']):raise ValueError('fixed source range keys differ')
            for leaf,interval in ranges.items():
                if (len(interval)!=2 or any(type(n)is not int for n in interval)
                    or not 0<=interval[0]<=interval[1]
                    or interval[1]-interval[0]!=self.plan['source_targets'][leaf]):
                    raise ValueError('fixed range differs from quota')
        def groups():
            group=[];identity=None
            for rows in self.prepared_record_batches(path):
                entries=self.ledger.get_many([r['_sha']for r in rows])
                for record in rows:
                    entry=entries.get(record['_sha'])
                    if (not record['whole_record_protected']or record['allocation']!=self.plan['fixed_allocation_id']
                        or entry!=(record['_row_sha'],record['leaf'],owner)):
                        raise ValueError('protected record differs from its ledger')
                    n=record['allocated_content_tokens'];offset=record.get('allocated_document_offset',0)
                    if not 0<=offset<record['original_content_tokens']or not 0<n<=record['original_content_tokens']-offset:
                        raise ValueError('invalid protected span')
                    base=observed[record['leaf']];observed[record['leaf']]+=n
                    if paired and record['source_text_sha256'] in paired.accepted:
                        if offset or n!=record['original_content_tokens']:
                            raise ValueError('paired whole record differs from its reserved span')
                        continue
                    if ranges is not None:
                        lo,hi=ranges[record['leaf']];begin,end=max(base,lo),min(base+n,hi)
                        if begin>=end:continue
                        offset+=begin-base;n=end-begin
                    new=(record['leaf'],record['stratum'],record['source_bin_uri'],record['source_bin_sha256'])
                    if group and new!=identity:
                        yield identity,group;group=[]
                    identity=new;group.append((record,offset,n))
            if group:yield identity,group
            expected=self.plan.get('fixed_total_source_targets')
            if expected is not None and dict(observed)!={leaf:n for leaf,n in expected.items() if n!=0}:
                raise ValueError('fixed total differs from reserved selection')
        pending=deque();self.leaf=self.stratum=None;self.buffer_number=0
        def submit(identity,records):
            spec={'uri':identity[2],'sha256':identity[3]}
            def receive():
                b,k=spec['uri'][5:].split('/',1);spec['bytes']=self.client.head_object(Bucket=b,Key=k)['ContentLength']
                target=self.input_path(spec,'.fixed.uint16');self.download(spec,target);return target
            return identity,records,self.io_pool.submit(receive)
        def consume(item):
            identity,records,future=item;target=future.result()
            if identity[:2]!=(self.leaf,self.stratum):
                self.flush(buffer);self.finish_stratum();self.leaf,self.stratum=identity[:2];self.buffer_number=0
            references=bool(self.plan.get('parallel_fragment_preparation'))
            tokens=None if references else np.memmap(target,dtype='<u2',mode='r')
            token_count=Path(target).stat().st_size//2 if references else len(tokens)
            try:
                for record,offset,n in records:
                    begin=record['source_token_start']+offset
                    if begin<0 or begin+n>token_count:raise ValueError('fixed span exceeds source payload')
                    document={k:v for k,v in record.items()if k not in
                        ('allocation','allocated_content_tokens','allocated_document_offset','whole_record_protected',
                         'leaf','stratum','_sha','_row_sha')}
                    complete=offset+n==record['original_content_tokens']
                    parts=(FragmentReference(str(target),begin,n,document,offset,complete,i)
                        for i in range((n+self.plan['sequence_length']-1)//self.plan['sequence_length']))if references else fragments(
                            tokens[begin:begin+n],document,length=self.plan['sequence_length'],offset=offset,document_end=complete)
                    for part in parts:
                        if references:self.retain_buffer_input(target)
                        buffer.append(part)
                        if len(buffer)>=self.plan['packing_buffer_fragments']:self.flush(buffer)
                    counts[record['leaf']]+=n
                    if counts[record['leaf']]>self.plan['source_targets'][record['leaf']]:
                        raise ValueError('fixed source exceeds declared quota')
            finally:del tokens;self.release(target)
        for identity,records in groups():
            pending.append(submit(identity,records))
            if len(pending)>=self.plan.get('payload_prefetch',2):consume(pending.popleft())
        while pending:consume(pending.popleft())
        self.flush(buffer);self.finish_stratum()
        return [{'leaf':leaf,'target':target,'selected':counts[leaf],'shortfall':target-counts[leaf]}
            for leaf,target in self.plan['source_targets'].items()]

    def run(self):
        try:
            if self.plan.get('predecessor_mixture_preparation'):
                from production.packed_mixture_finalize import prepare_phase
                preparation=self.plan['predecessor_mixture_preparation']|{
                    'expected_input_ledger':self.plan['input_ledger']}
                prepared=prepare_phase(self.client,preparation,
                    self.work/'predecessor-mixture')
                self.event('predecessor_mixture_ready',**prepared)
            results=self.materialize_fixed()if self.plan.get('fixed_selection')else self.select_sources()
            self.finish_stratum(force=True);self.provenance.close()
            if self.reserved:self.reserved.close()
            checkpoint=None if self.readonly_ledger else self.ledger.snapshot(self.work/'selected.sqlite')
            mode='immutable-predecessor'if self.readonly_ledger else 'ram'if self.ledger.memory else 'bounded-file-fallback'
            count=self.ledger.count
            self.ledger.close()
            self.ledger=None
            suffix=self.work/'quota-suffix-reservations.jsonl'
            suffix.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in self.pending_suffixes))
            artifacts={'selected.sqlite':dict(self.plan['input_ledger'])}if self.readonly_ledger else {}
            for item in [self.events,checkpoint,suffix,self.reserved_path,self.work/'TRANSLATION_PAIRS.json']:
                if item is not None and item.exists():
                    artifact=publish(self.client,self.bucket,item,self.prefix+'/audit/'+item.name)
                    name=item.name
                    if name=='protected-records.jsonl.gz':
                        name='protected-records.jsonl';artifact['compression']='gzip'
                    artifacts[name]=artifact
            archive=shutil.make_archive(str(self.work/'source-receipts'),'gztar',self.source_receipts_dir)
            publish(self.client,self.bucket,Path(archive),self.prefix+'/audit/source-receipts.tar.gz')
            report={'schema':'terminus_warmup_materialization_v1','plan_sha256':self.plan['plan_sha256'],
                'results':results,'totals':dict(self.totals),'shards':self.receipts,
                'physical_creation_finished':not self.reservation_only,'reservation_creation_finished':self.reservation_only,
                'allocation_id':self.allocation,'artifacts':artifacts,'input_ledger':self.plan.get('input_ledger'),
                'all_curriculum_quotas_satisfied':all(r['shortfall']==0 for r in results),
                'cross_document_mask_required':True,'training_manifest_published':False,'native_training_accepted':False,
                'execution':{'worker':'fast_packing_worker','ledger_schema':'ci1_compact_ledger_v2',
                    'working_ledger_storage':mode,'ledger_records':count,
                    'metadata_processes':self.plan.get('metadata_workers',4),
                    'packing_processes':self.plan.get('packing_workers',8)}}
            if hasattr(self,'pair_report'):report['translation_pairs']=self.pair_report
            if not self.reservation_only and report['all_curriculum_quotas_satisfied']:
                from production.packed_mixture import mixture_candidate
                mixture=self.work/'MIXTURE_CANDIDATE.json'
                mixture.write_text(json.dumps(mixture_candidate(report),indent=2)+'\n')
                artifacts[mixture.name]=publish(self.client,self.bucket,mixture,self.prefix+'/audit/'+mixture.name)
            target=self.work/'MATERIALIZATION.json';target.write_text(json.dumps(report,indent=2)+'\n')
            publish(self.client,self.bucket,target,self.prefix+'/MATERIALIZATION.json')
            self.event('finished',quotas_satisfied=report['all_curriculum_quotas_satisfied']);return report
        finally:
            self.io_pool.shutdown(wait=True,cancel_futures=True)
            self.metadata_pool.shutdown(wait=True,cancel_futures=True)
            self.packing_pool.shutdown(wait=True,cancel_futures=True)
            self.upload_pool.shutdown(wait=True,cancel_futures=True)
            if self.ledger is not None:self.ledger.close()
            if not self.provenance.closed:self.provenance.close()
            if self.reserved and not self.reserved.closed:self.reserved.close()
            shutil.rmtree(self.scratch,ignore_errors=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--work',required=True);p.add_argument('--profile')
    a=p.parse_args();path=Path(a.plan);plan=json.loads(path.read_text());plan['plan_sha256']=file_sha(path)
    for name,expected in [('jobs_path','jobs_sha256'),('exclusions_path','exclusions_sha256')]:
        if file_sha(plan[name])!=plan[expected]:raise ValueError('plan input hash differs')
    worker=FastWorker(plan,a.work,a.profile)
    try:worker.run()
    except BaseException as exc:worker.event('failed',error=type(exc).__name__+': '+str(exc));raise


if __name__=='__main__':main()
