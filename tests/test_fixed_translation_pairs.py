from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from botocore.exceptions import ClientError

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from production.compact_ledger import CompactLedger
from production.document_packing import file_sha
from production.fast_packing_worker import FastWorker
from production.fixed_translation_pairs import SourceMetadata, family_partition, validate_policy
from production.translation_pair_policy import validate_program
from production.packed_mixture import PackedMixture, PackedSequenceReader
from tests.test_fast_packing_worker import publish
from tests.test_packing_worker import FakeS3, fake_publish
from tests.test_translation_pairs import Encoder
from terminus.tokenizer65k import ATOM_REVISION_SHA256


class RangedS3(FakeS3):
    def __init__(self):super().__init__();self.range_requests=0
    def head_object(self,**kwargs):
        return super().head_object(**kwargs)|{'ETag':'"'+hashlib.md5(self.objects[(kwargs['Bucket'],kwargs['Key'])]).hexdigest()+'"'}
    def get_object(self,Bucket,Key,Range=None,IfMatch=None):
        raw=self.objects[(Bucket,Key)]
        if IfMatch is not None:
            if IfMatch!=self.head_object(Bucket=Bucket,Key=Key)['ETag']:raise ValueError('snapshot changed')
        if Range:
            self.range_requests+=1;begin,end=map(int,Range[6:].split('-'));raw=raw[begin:end+1]
        return {'Body':io.BytesIO(raw)}


def policy(cap=1000):
    return dict(schema='ci1_fixed_translation_pairs_v1',phase='ordinary_dense_stable',
        program_ceiling_content_tokens=20_000_000_000,phase_ceiling_content_tokens=cap,
        family_partition_count=1,family_partition_index=0,family_seed='one-pair-per-family',
        minimum_translation_score=95,max_preparation_seconds=60)


def fixture(root):
    client=RangedS3();encoder=Encoder();records=[];jobs=[];targets=Counter()
    ex=root/'exclusions';ex.write_text('')
    for language in ('en','hi'):
        raw=[];index=[];values=[]
        source='datasets/Nemotron-Pretraining-Specialized-v1.1/Code-Concepts/part_000.parquet'if language=='en'else'datasets/nemotron-indic-specialized-v1/hi-Deva-IN/Code-Concepts/part.parquet'
        for i in range(2):
            seed='The answer is 42.'if i==0 else'The answer is 43.'
            text=seed if language=='en'else('उत्तर ४२ है।'if i==0 else'उत्तर ४३ है।')
            uid='origin-'+str(i);m=dict(text=text,uuid=uid if language=='en'else'hi-'+uid)
            if language!='en':m.update(source_uuid=uid,source_text=seed,language_code='hi-Deva-IN',
                overall_score=98 if i==0 else 92,evaluation_result={'error_analysis':[]},
                _source_repo='nvidia/Nemotron-Pretraining-Specialized-v1.1',_source_config='Code-Concepts',_source_file='Code-Concepts/part_000.parquet')
            raw.append(m);tokens=encoder.encode(text)
            r=dict(doc_id=m['uuid'],row_key=source+':'+str(i),split_family_id=uid,
                source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),n_tokens=len(tokens),
                token_start=len(values),validation=False)
            index.append(r);values.extend(tokens)
        rawpath=root/(language+'.parquet');pq.write_table(pa.Table.from_pylist(raw),rawpath)
        client.objects[('bucket',source)]=rawpath.read_bytes()
        bp=root/(language+'.bin');bp.write_bytes(np.array(values,dtype='<u2').tobytes())
        ip=root/(language+'.index');ip.write_text(''.join(json.dumps(r)+'\n'for r in index))
        binary=fake_publish(client,'bucket',bp,language+'/bin');ix=fake_publish(client,'bucket',ip,language+'/index')
        for r in index:
            leaf=language+'/Code';n=r['n_tokens'];targets[leaf]+=n
            records.append({k:r[k]for k in ('doc_id','row_key','split_family_id','source_text_sha256')}|
                dict(source_bin_uri=binary['uri'],source_bin_sha256=binary['sha256'],source_index_sha256=ix['sha256'],
                    source_token_start=r['token_start'],original_content_tokens=n,allocated_content_tokens=n,
                    allocation='dense',whole_record_protected=True,leaf=leaf,stratum=language+'/concepts'))
        head=client.head_object(Bucket='bucket',Key=source)
        job=dict(job_id=language,source_key=source,source_snapshot={'bytes':head['ContentLength'],'etag':head['ETag']},
            text_columns=['text'],parts=[dict(index_key=language+'/old',validation=False,token_snapshot={'bytes':len(values)*2})])
        jobs.append(job)
        result=dict(status='verified',source_key=source,job_id=language,original_holdout_assignments_preserved=True,
            encoder_identity=dict(tokenizer_sha256=ATOM_REVISION_SHA256,roundtrip_checked_per_document=True),
            outputs=[dict(original_index_key=language+'/old',validation=False,tokens=len(values),documents=2,bin=binary,index=ix)])
        client.put_object(Bucket='bucket',Key='converted/progress/'+language+'.json',Body=json.dumps(result).encode())
    ledger=CompactLedger(root/'ledger',memory_mib=16,seen_mib=4,cache_mib=4)
    for r in records:ledger.add_many([r],r['leaf'],'protected:dense')
    lp=ledger.snapshot(root/'selected.sqlite');ledger.close();ledger_spec=fake_publish(client,'bucket',lp,'ledger')
    fixed=root/'fixed';fixed.write_text(''.join(json.dumps(r)+'\n'for r in records));fixed_spec=fake_publish(client,'bucket',fixed,'fixed')
    jp=root/'jobs';jp.write_text(''.join(json.dumps(j)+'\n'for j in jobs))
    plan=dict(bucket='bucket',exclusions_path=str(ex),exclusions_sha256=file_sha(ex),
        deadline_unix=time.time()+600,conversion_prefixes=['converted'],tokenizer_sha256=ATOM_REVISION_SHA256,
        plan_sha256='plan',seed='paired',sequence_length=128,sequences_per_shard=2,
        packing_buffer_fragments=2,jobs_path=str(jp),scratch_directory=str(root/'ram'),
        ledger_memory_mib=16,ledger_seen_mib=4,ledger_fallback_cache_mib=4,input_ram_mib=16,
        metadata_workers=1,packing_workers=1,metadata_prefetch=2,payload_prefetch=2,
        packing_prefetch=2,upload_prefetch=2,source_targets=dict(targets),input_ledger=ledger_spec,
        fixed_selection=fixed_spec,fixed_allocation_id='dense',fixed_total_source_targets=dict(targets),
        output_prefix='paired',translation_pairs=policy(),tokenizer_path='fake-pinned-artifact')
    return client,plan,records


class FixedPairTests(unittest.TestCase):
    def test_unavailable_optional_metadata_keeps_all_owned_records_ordinary(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);client,plan,records=fixture(root)
            with patch('production.fast_packing_worker.boto3.Session')as session,\
                 patch('production.fast_packing_worker.publish',side_effect=publish),\
                 patch('terminus.tokenizer_native.Native65kTokenizer',return_value=Encoder()),\
                 patch('production.fixed_translation_pairs.SourceMetadata.get',side_effect=ClientError(
                    {'Error':{'Code':'AccessDenied','Message':'metadata unavailable'}},'GetObject')):
                session.return_value.client.return_value=client
                report=FastWorker(plan,root/'ordinary-fallback').run()
            self.assertTrue(report['all_curriculum_quotas_satisfied'])
            self.assertEqual(report['translation_pairs']['pair_count'],0)
            self.assertEqual(report['translation_pairs']['removed_ordinary_records'],0)
            self.assertEqual(report['totals']['content_tokens'],sum(plan['source_targets'].values()))
            self.assertFalse(report['totals'].get('synthetic_tokens',0))

    def test_materialization_replaces_both_owned_occurrences_and_reader_counts_both_sources(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);client,plan,records=fixture(root)
            with patch('production.fast_packing_worker.boto3.Session')as session,\
                 patch('production.fast_packing_worker.publish',side_effect=publish),\
                 patch('terminus.tokenizer_native.Native65kTokenizer',return_value=Encoder()):
                session.return_value.client.return_value=client
                report=FastWorker(plan,root/'worker').run()
            self.assertTrue(report['all_curriculum_quotas_satisfied'])
            self.assertEqual(report['translation_pairs']['pair_count'],1)
            self.assertEqual(report['translation_pairs']['removed_ordinary_records'],2)
            self.assertEqual(report['totals']['content_tokens'],sum(plan['source_targets'].values()))
            self.assertEqual(report['totals']['synthetic_tokens'],2)
            candidate=json.loads((root/'worker/MIXTURE_CANDIDATE.json').read_text())
            self.assertEqual(candidate['source_content_tokens'],plan['source_targets'])
            paths={}
            for shard in candidate['shards']:
                for name in ['payload','index','index_offsets']:
                    spec=shard[name];p=root/(spec['sha256']+name)
                    p.write_bytes(client.objects[('bucket',spec['uri'][len('s3://bucket/'):])]);paths[spec['uri']]=p
            reader=PackedSequenceReader(PackedMixture(candidate['shards'],seed='test'),resolve_path=lambda uri:paths[uri])
            counts=Counter();pair_samples=0
            for i in range(reader.mixture.total_sequences):
                sample=reader.sample(i)
                for c in sample['source_content_tokens']:counts[c['leaf']]+=c['tokens']
                if sample['synthetic_tokens']:
                    pair_samples+=1;self.assertEqual(len(set(sample['segment_ids'][sample['segment_ids']>=0])),1)
            self.assertEqual(dict(counts),plan['source_targets']);self.assertEqual(pair_samples,1)

    def test_source_cache_reuses_rows_and_snapshot_change_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);client,plan,records=fixture(root)
            worker=type('Worker',(),{'client':client,'bucket':'bucket'})()
            jobs={j['source_key']:j for j in map(json.loads,Path(plan['jobs_path']).read_text().splitlines())}
            cache=SourceMetadata(worker,jobs)
            self.assertEqual(cache.get(records[0])[1]['text'],'The answer is 42.')
            before=client.range_requests;cache.get(records[1]);self.assertEqual(client.range_requests,before)
            job=next(iter(jobs.values()));client.objects[('bucket',job['source_key'])]=b'changed'
            with self.assertRaises(ValueError):SourceMetadata(worker,jobs).get(records[0])

    def test_global_family_owner_and_policy_bounds(self):
        self.assertEqual(family_partition('uuid',14,'seed'),family_partition('uuid',14,'seed'))
        validate_policy(policy())
        for changes in ({'phase':'qat_stable'},{'phase_ceiling_content_tokens':20_000_000_001},
            {'max_preparation_seconds':1801},{'family_partition_index':1}):
            with self.assertRaises(ValueError):validate_policy(policy()|changes)

    def test_program_cannot_multiply_cap_across_workers_or_pair_qat(self):
        tasks=[]
        for i in range(14):
            identifier=f'dense-stable-{i+1:02d}'
            p=policy(20_000_000_000//14+int(i<20_000_000_000%14))
            p.update(family_partition_count=14,family_partition_index=i)
            tasks.append({'id':'materialize-'+identifier,'kind':'packing',
                'fixed_selection_from':identifier,'plan':{'translation_pairs':p}})
        validate_program(tasks)
        self.assertEqual(sum(t['plan']['translation_pairs']['phase_ceiling_content_tokens']for t in tasks),20_000_000_000)
        tasks[0]['plan']['translation_pairs']['phase_ceiling_content_tokens']+=1
        with self.assertRaises(ValueError):validate_program(tasks)
        tasks[0]['plan']['translation_pairs']['phase_ceiling_content_tokens']-=1
        tasks[0]['fixed_selection_from']='reserve-qat-finishing'
        with self.assertRaises(ValueError):validate_program(tasks)


if __name__=='__main__':unittest.main()
