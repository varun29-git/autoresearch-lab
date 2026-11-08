"""Bounded pairing of whole records already owned by a dense reservation.

The existing allocation ledger remains authoritative. Discovery IDs are hints;
raw source lineage, actual token spans and heldout exclusions qualify each pair.
Ordinary materialization omits exactly the accepted whole-record occurrences.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import threading
import time

from botocore.exceptions import BotoCoreError, ClientError

import numpy as np

from production.compact_ledger import key, row_key_hash
from production.native_corpus import _output_matches, _source_text
from production.translation_pairs import Passage, PairBudget, compound_pair
from production.translation_pair_policy import validate_policy
from terminus.tokenizer65k import ATOM_REVISION_SHA256

SCRIPTS={'hi':'Deva','bn':'Beng','gu':'Gujr','kn':'Knda','mr':'Deva','ta':'Taml','te':'Telu'}
ORIGINALS={'nvidia/Nemotron-Pretraining-Specialized-v1.1':
           'datasets/Nemotron-Pretraining-Specialized-v1.1/'}


def family_partition(uid, count, seed):
    return int.from_bytes(hashlib.sha256((seed+'\x1f'+str(uid)).encode()).digest()[:4],'big')%count


class RangeFile(io.RawIOBase):
    """Seekable S3 ranges pinned to the original source ETag."""
    def __init__(self, client, bucket, name, snapshot):
        super().__init__();self.client=client;self.bucket=bucket;self.name=name
        self.size=snapshot['bytes'];self.etag=snapshot['etag'];self.position=0
        head=client.head_object(Bucket=bucket,Key=name)
        if head['ETag']!=self.etag or head['ContentLength']!=self.size:
            raise ValueError('original source snapshot changed')

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        new=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if new<0:raise ValueError('invalid source seek')
        self.position=new;return new

    def read(self,n=-1):
        if n is None or n<0:n=self.size-self.position
        n=min(n,max(0,self.size-self.position))
        if not n:return b''
        body=self.client.get_object(Bucket=self.bucket,Key=self.name,
            Range=f'bytes={self.position}-{self.position+n-1}',IfMatch=self.etag)['Body']
        try:raw=body.read()
        finally:body.close()
        if len(raw)!=n:raise ValueError('truncated original source range')
        self.position+=n;return raw

    def readinto(self,buffer):
        raw=self.read(len(buffer));buffer[:len(raw)]=raw;return len(raw)


class SourceMetadata:
    """Bounded Arrow row-group cache; convert only requested rows to Python."""
    def __init__(self,worker,jobs,cache_bytes=2<<30):
        self.worker=worker;self.jobs=jobs;self.cache=OrderedDict();self.bytes=0
        self.limit=cache_bytes;self.lock=threading.Lock();self.layouts={};self.source_locks={}

    def get(self,record):
        import pyarrow.parquet as pq
        source,number=record['row_key'].rsplit(':',1);number=int(number)
        job=self.jobs.get(source)
        if (not job or job.get('source_format','parquet')!='parquet'
                or job.get('renderer','field')!='field' or job.get('text_columns',['text'])!=['text']):
            raise ValueError('pair source needs an approved original Parquet text field')
        with self.lock:source_lock=self.source_locks.setdefault(source,threading.Lock())
        # Single-flight row-group reads and cached footer geometry avoid a
        # HEAD/footer request for every matched document in the same file.
        with source_lock:
            if source not in self.layouts:
                with RangeFile(self.worker.client,self.worker.bucket,source,job['source_snapshot'])as handle:
                    parquet=pq.ParquetFile(handle)
                    self.layouts[source]=[parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
            base=0;group=None
            for i,n in enumerate(self.layouts[source]):
                if base<=number<base+n:group=i;break
                base+=n
            if group is None:raise ValueError('pair row exceeds original source')
            cache_key=(source,group,job['source_snapshot']['etag'])
            with self.lock:
                table=self.cache.get(cache_key)
                if table is not None:self.cache.move_to_end(cache_key)
            if table is None:
                with RangeFile(self.worker.client,self.worker.bucket,source,job['source_snapshot'])as handle:
                    parquet=pq.ParquetFile(handle);names=set(parquet.schema_arrow.names)
                    columns=[c for c in ('text','uuid','source_uuid','source_text','language_code',
                        'overall_score','evaluation_result','_source_repo','_source_config','_source_file')if c in names]
                    table=parquet.read_row_group(group,columns=columns,use_threads=False)
                if table.nbytes<=self.limit:
                    with self.lock:
                        previous=self.cache.pop(cache_key,None)
                        if previous is not None:self.bytes-=previous.nbytes
                        self.cache[cache_key]=table;self.bytes+=table.nbytes
                        while self.bytes>self.limit:
                            _,old=self.cache.popitem(last=False);self.bytes-=old.nbytes
            value=table.slice(number-base,1).to_pylist()[0]
        if _source_text(value,job).encode('utf-8')!=value['text'].encode('utf-8'):
            raise ValueError('pair source rendering differs')
        return job,value


class FixedPairPreparation:
    def __init__(self,worker,selection,policy):
        validate_policy(policy)
        if (not worker.plan.get('fixed_selection') or worker.plan.get('fixed_source_ranges')
                or worker.reservation_only or worker.plan['tokenizer_sha256']!=ATOM_REVISION_SHA256):
            raise ValueError('pairing requires a complete dense fixed reservation')
        self.worker=worker;self.selection=Path(selection);self.policy=policy
        self.stop=time.monotonic()+policy['max_preparation_seconds']
        self.accepted=set();self.charges=Counter();self.stratum_charges=Counter();self.rejections=Counter()
        self.completed_families=set()
        self.pair_count=0;self.synthetic_tokens=0;self.budget=None
        self.jobs={}
        with Path(worker.plan['jobs_path']).open()as source:
            for line in source:
                job=json.loads(line);self.jobs[job['source_key']]=job
        self.metadata=SourceMetadata(worker,self.jobs)
        self.verified_jobs={};self.bin_heads={}

    def candidates(self):
        import duckdb
        if duckdb.__version__!='1.4.2':raise ValueError('pair discovery requires DuckDB 1.4.2')
        conn=duckdb.connect(str(self.worker.work/'pair-candidates.duckdb'),
            config={'memory_limit':'6GB','threads':4})
        columns={k:'VARCHAR'for k in ('doc_id','row_key','split_family_id','source_text_sha256',
            'source_bin_uri','source_bin_sha256','source_index_sha256','leaf','stratum','allocation')}
        columns.update({k:'BIGINT'for k in ('source_token_start','original_content_tokens',
            'allocated_content_tokens','allocated_document_offset')});columns['whole_record_protected']='BOOLEAN'
        interrupted=threading.Event()
        def interrupt():interrupted.set();conn.interrupt()
        timer=threading.Timer(max(0.01,self.stop-time.monotonic()),interrupt);timer.daemon=True;timer.start()
        remaining=self.worker.plan['source_targets']
        try:
            self.budget=PairBudget({(l,s):n for l,s,n in conn.execute("""SELECT leaf,stratum,SUM(allocated_content_tokens)
                FROM read_json(?,format='newline_delimited',columns=?) GROUP BY leaf,stratum""",
                [str(self.selection),columns]).fetchall()},ceiling=self.policy['phase_ceiling_content_tokens'])
        except duckdb.InterruptException:
            timer.cancel();conn.close()
            if not interrupted.is_set():raise
            self.rejections['preparation_time_budget']+=1;return
        except BaseException:
            timer.cancel();conn.close();raise
        actual=Counter()
        for (l,s),n in self.budget.remaining.items():actual[l]+=n
        if dict(actual)!=remaining:
            timer.cancel();conn.close();raise ValueError('pair discovery source quotas differ from reserved selection')
        try:
            conn.execute("""CREATE TABLE eligible AS
                SELECT *,split_part(leaf,'/',1) AS language FROM read_json($source,format='newline_delimited',columns=$columns)
                WHERE whole_record_protected=true AND allocated_content_tokens=original_content_tokens
                AND COALESCE(allocated_document_offset,0)=0 AND original_content_tokens BETWEEN 1 AND 4093
                AND (('0x'||substr(sha256($seed||chr(31)||split_family_id),1,8))::UBIGINT % $parts)=$part""",
                {'source':str(self.selection),'columns':columns,'seed':self.policy['family_seed'],
                 'parts':self.policy['family_partition_count'],'part':self.policy['family_partition_index']})
            cursor=conn.execute("""WITH variants AS (
                SELECT *,ROW_NUMBER() OVER (PARTITION BY split_family_id,language ORDER BY source_text_sha256,row_key) AS variant
                FROM eligible WHERE language='en' OR regexp_matches(row_key,'/(hi-Deva|bn-Beng|gu-Gujr|kn-Knda|mr-Deva|ta-Taml|te-Telu)-IN/')),
                languages AS (SELECT *,ROW_NUMBER() OVER (PARTITION BY split_family_id ORDER BY sha256(split_family_id||language||?),language) AS side
                FROM variants WHERE variant=1)
                SELECT to_json(a),to_json(b) FROM languages a JOIN languages b USING(split_family_id)
                WHERE a.side<b.side AND a.side<=3 AND b.side<=4
                AND a.allocated_content_tokens+b.allocated_content_tokens+2<=4096
                ORDER BY a.row_key,b.row_key""",[self.policy['family_seed']])
            while time.monotonic()<self.stop:
                rows=cursor.fetchmany(32)
                if not rows:break
                yield [(json.loads(a),json.loads(b))for a,b in rows]
        except duckdb.InterruptException:
            if not interrupted.is_set():raise
            self.rejections['preparation_time_budget']+=1
        finally:
            timer.cancel();conn.close()

    def record_tokens(self,record):
        owner='protected:'+self.worker.plan['fixed_allocation_id']
        if (record['allocation']!=self.worker.plan['fixed_allocation_id']
                or self.worker.ledger.get(record['source_text_sha256'])!=(
                    row_key_hash(record['row_key']),record['leaf'],owner)):
            raise ValueError('pair record differs from the current dense ownership ledger')
        spec={'uri':record['source_bin_uri'],'sha256':record['source_bin_sha256']}
        b,k=spec['uri'][5:].split('/',1)
        if spec['uri']not in self.bin_heads:
            head=self.worker.client.head_object(Bucket=b,Key=k,ChecksumMode='ENABLED')
            spec['bytes']=head['ContentLength']
            if not _output_matches(head,spec):raise ValueError('pair token payload identity differs')
            self.bin_heads[spec['uri']]=head
        head=self.bin_heads[spec['uri']];start=record['source_token_start']*2
        size=record['original_content_tokens']*2
        if start<0 or start+size>head['ContentLength']:raise ValueError('pair span exceeds source payload')
        body=self.worker.client.get_object(Bucket=b,Key=k,Range=f'bytes={start}-{start+size-1}',IfMatch=head['ETag'])['Body']
        try:raw=body.read()
        finally:body.close()
        if len(raw)!=size:raise ValueError('truncated pair token span')
        return np.frombuffer(raw,dtype='<u2')

    def qualify(self,records,metadata,tokenizer):
        passages=[];tokens=[]
        for side,(record,(job,value))in enumerate(zip(records,metadata)):
            if job['job_id']not in self.verified_jobs:
                result=self.worker.receipt(job)
                if result is None:raise ValueError('missing verified source conversion for pair')
                outputs={(o['bin']['sha256'],o['index']['sha256'])for o in result['outputs']if not o['validation']}
                if (record['source_bin_sha256'],record['source_index_sha256'])not in outputs:
                    raise ValueError('pair source differs from its conversion receipt')
                self.worker.provenance_receipt(job,result);self.verified_jobs[job['job_id']]=outputs
            if (record['source_bin_sha256'],record['source_index_sha256'])not in self.verified_jobs[job['job_id']]:
                raise ValueError('pair source differs from its conversion receipt')
            value=dict(value);language=record['language'];original=language=='en'
            if not original and value.get('language_code')!=language+'-'+SCRIPTS.get(language,'')+'-IN':
                raise ValueError('pair needs a qualified native-script source')
            if original:
                other=metadata[1-side][1]
                repo=other.get('_source_repo');prefix=ORIGINALS.get(repo)
                if (not prefix or not job['source_key'].startswith(prefix)
                        or not job['source_key'].endswith('/'+str(other.get('_source_file')))
                        or str(value.get('uuid'))!=str(other.get('source_uuid'))):
                    raise ValueError('English counterpart lineage is not source-bound')
                for name in ('_source_repo','_source_config','_source_file'):value[name]=other[name]
            adapted=record|{'n_tokens':record['original_content_tokens'],'validation':False}
            passages.append(Passage(adapted,value,language,record['leaf'],record['stratum'],original))
            tokens.append(self.record_tokens(record))
        pair=compound_pair(*passages,*tokens,tokenizer=tokenizer,exclusions=self.worker.exclusions,
            length=self.worker.plan['sequence_length'],minimum_score=95)
        uid=pair.document['lineage'][3]
        if family_partition(uid,self.policy['family_partition_count'],self.policy['family_seed'])!=self.policy['family_partition_index']:
            raise ValueError('original family belongs to another dense tranche')
        return pair

    def pairs(self):
        from terminus.tokenizer_native import Native65kTokenizer
        tokenizer=Native65kTokenizer(self.worker.plan['tokenizer_path'],expected_sha256=ATOM_REVISION_SHA256)
        with ThreadPoolExecutor(max_workers=6)as pool:
            for batch in self.candidates():
                batch=[records for records in batch if records[0]['split_family_id']not in self.completed_families]
                flattened=[r for pair in batch for r in pair]
                futures=[pool.submit(self.metadata.get,r)for r in flattened]
                for i,records in enumerate(batch):
                    if time.monotonic()>=self.stop:return
                    if records[0]['split_family_id']in self.completed_families:continue
                    try:
                        metadata=[futures[2*i+j].result()for j in range(2)]
                        pair=self.qualify(records,metadata,tokenizer)
                    except (ValueError, BotoCoreError, ClientError) as exc:
                        self.rejections[str(exc)]+=1;continue
                    if not self.budget.claim(pair,phase='dense-stable'):
                        self.rejections['phase_ceiling_or_source_quota']+=1
                        if self.budget.ceiling-self.budget.content_tokens<2:return
                        continue
                    self.accepted.update(r['source_text_sha256']for r in records)
                    self.completed_families.add(records[0]['split_family_id'])
                    for r in records:
                        n=r['original_content_tokens'];self.charges[r['leaf']]+=n
                        self.stratum_charges[(r['leaf'],r['stratum'])]+=n
                    self.pair_count+=1;self.synthetic_tokens+=pair.synthetic_tokens
                    yield pair
                    self.worker.deadline()

    def report(self):
        return {'schema':'ci1_fixed_translation_pairs_v1','policy':self.policy,
            'pair_count':self.pair_count,'paired_content_tokens':sum(self.charges.values()),
            'synthetic_tokens':self.synthetic_tokens,'removed_ordinary_records':len(self.accepted),
            'source_content_tokens':[{'leaf':l,'stratum':s,'tokens':n}for(l,s),n in sorted(self.stratum_charges.items())],
            'rejections':dict(self.rejections),'native_training_accepted':False}
