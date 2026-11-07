import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from production.compact_ledger import CompactLedger, key, row_key_hash
from production.document_packing import file_sha
from production.fast_packing_worker import FastWorker
from production.packing_worker import Worker, fingerprint
from tests.test_packing_worker import FakeS3, fake_publish


def publish(client,bucket,path,name,**kwargs):
    if kwargs.get('expected_sha256'):
        if file_sha(path)!=kwargs['expected_sha256']:raise ValueError('wrong publication digest')
    return fake_publish(client,bucket,path,name)


def fixture(root):
    client=FakeS3();ex=root/'exclusions';ex.write_text(json.dumps({'family_id':'frozen',
        'locations':[{'doc_id':'held','row_key':'held'}]})+'\n');jobs=[]
    for j,leaf in enumerate(('en/source','hi/source')):
        rows=[];values=[]
        for i in range(60):
            n=(1,7,8,10,23)[i%5];sha=hashlib.sha256(('common'if i in(2,3)else str(j)+':'+str(i)).encode()).hexdigest()
            row=dict(doc_id='held'if i==0 else 'दस्तावेज-'+str(i),row_key='café-🚀-'+str(j)+':'+str(i),
                split_family_id='family-'+str(j)+':'+str(i),source_text_sha256=sha,validation=False,
                token_start=len(values),n_tokens=n)
            rows.append(row);values.extend(range(100+j*1000+i*25,100+j*1000+i*25+n))
        bp=root/('bin'+str(j));bp.write_bytes(np.asarray(values,dtype='<u2').tobytes())
        ip=root/('index'+str(j));ip.write_text(''.join(json.dumps(r)+'\n'for r in rows))
        binary=fake_publish(client,'bucket',bp,'input/bin'+str(j));index=fake_publish(client,'bucket',ip,'input/index'+str(j))
        job=dict(job_id='job'+str(j),source_key='raw'+str(j),packing_leaf=leaf,packing_stratum=leaf,
            parts=[dict(index_key='original'+str(j),validation=False,token_snapshot={'bytes':len(values)*2})])
        receipt=dict(status='verified',source_key=job['source_key'],job_id=job['job_id'],
            original_holdout_assignments_preserved=True,
            encoder_identity=dict(tokenizer_sha256='tok',roundtrip_checked_per_document=True),
            outputs=[dict(original_index_key='original'+str(j),validation=False,tokens=len(values),documents=len(rows),bin=binary,index=index)])
        client.put_object(Bucket='bucket',Key='converted/progress/'+job['job_id']+'.json',Body=json.dumps(receipt).encode())
        jobs.append(job)
    jp=root/'jobs';jp.write_text(''.join(json.dumps(j)+'\n'for j in jobs))
    plan=dict(bucket='bucket',exclusions_path=str(ex),exclusions_sha256=file_sha(ex),
        deadline_unix=time.time()+600,conversion_prefixes=['converted'],tokenizer_sha256='tok',
        plan_sha256='plan',seed='equivalence',sequence_length=8,sequences_per_shard=5,
        packing_buffer_fragments=5,jobs_path=str(jp),scratch_directory=str(root/'ram'),
        ledger_memory_mib=32,ledger_seen_mib=8,ledger_fallback_cache_mib=8,input_ram_mib=16,
        metadata_workers=2,packing_workers=2,metadata_prefetch=2,payload_prefetch=2,
        packing_prefetch=3,upload_prefetch=2)
    return client,plan


class FastPackingTests(unittest.TestCase):
    def test_corrupt_prefetched_payload_cannot_publish_completion(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);client,plan=fixture(root)
            client.objects[('bucket','input/bin0')]=b'broken bytes'
            with patch('production.fast_packing_worker.boto3.Session')as session, \
                 patch('production.fast_packing_worker.publish',side_effect=publish):
                session.return_value.client.return_value=client
                with self.assertRaisesRegex(ValueError,'identity differs'):
                    FastWorker(plan|{'output_prefix':'corrupt','source_targets':{'en/source':220}},root/'corrupt').run()
            self.assertNotIn(('bucket','corrupt/MATERIALIZATION.json'),client.objects)
            self.assertFalse((root/'corrupt/selected.sqlite').exists())

    def test_reservation_matches_reference_record_order_and_spans(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);client,plan=fixture(root)
            plan|={'source_targets':{'en/source':220,'hi/source':220},'reservation_only':True,'allocation_id':'matched'}
            with patch('production.packing_worker.boto3.Session')as session, \
                 patch('production.fast_packing_worker.boto3.Session')as fast_session, \
                 patch('production.packing_worker.publish',side_effect=publish), \
                 patch('production.fast_packing_worker.publish',side_effect=publish):
                session.return_value.client.return_value=client;fast_session.return_value.client.return_value=client
                reference=Worker(plan|{'output_prefix':'reference'},root/'reference').run()
                fast=FastWorker(plan|{'output_prefix':'fast'},root/'fast').run()
            self.assertEqual(reference['totals'],fast['totals'])
            # The fast path checks membership in batches; rejection counters
            # may include prefetched rows beyond the quota boundary.
            self.assertEqual([{k:v for k,v in r.items()if k!='rejections'}for r in reference['results']],
                             [{k:v for k,v in r.items()if k!='rejections'}for r in fast['results']])
            records=[]
            for name in ('reference','fast'):
                records.append([json.loads(line)for line in(root/name/'protected-records.jsonl').read_text().splitlines()])
            self.assertEqual(*records)

    def test_compact_checkpoint_migrates_identity_and_rejects_inconsistent_snapshot(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);old=root/'old.sqlite';db=sqlite3.connect(old)
            db.execute('CREATE TABLE selected(text_sha TEXT PRIMARY KEY,row_key TEXT,leaf TEXT,allocation TEXT)')
            sha=hashlib.sha256(b'source').hexdigest();db.execute('INSERT INTO selected VALUES(?,?,?,?)',
                (sha,'long Unicode row café','en/a','protected:qat'));db.commit();db.close()
            ledger=CompactLedger(root/'working',old,memory_mib=1,seen_mib=1,cache_mib=1)
            self.assertEqual(ledger.get(sha),(row_key_hash('long Unicode row café'),'en/a','protected:qat'))
            self.assertEqual(ledger.contains_many([key(sha)]),{key(sha)})
            checkpoint=ledger.snapshot(root/'checkpoint.sqlite');ledger.close()
            restored=CompactLedger(root/'restored',checkpoint,memory_mib=0,seen_mib=0,cache_mib=1)
            self.assertFalse(restored.memory)
            self.assertEqual(restored.get(sha),(row_key_hash('long Unicode row café'),'en/a','protected:qat'))
            restored.count+=1
            with self.assertRaises(ValueError):restored.snapshot(root/'invalid.sqlite')
            self.assertFalse((root/'invalid.sqlite').exists());restored.close()

    def test_fixed_materialization_matches_reference_from_legacy_protected_ledger(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);client,plan=fixture(root)
            plan|={'source_targets':{'en/source':220,'hi/source':220},'allocation_id':'matched'}
            with patch('production.packing_worker.boto3.Session')as session, \
                 patch('production.fast_packing_worker.boto3.Session')as fast_session, \
                 patch('production.packing_worker.publish',side_effect=publish), \
                 patch('production.fast_packing_worker.publish',side_effect=publish):
                session.return_value.client.return_value=client;fast_session.return_value.client.return_value=client
                reserve=Worker(plan|{'output_prefix':'reserve','reservation_only':True},root/'reserve').run()
                fixed=plan|dict(input_ledger=reserve['artifacts']['selected.sqlite'],
                    fixed_selection=reserve['artifacts']['protected-records.jsonl'],fixed_allocation_id='matched',
                    fixed_total_source_targets={'en/source':220,'hi/source':220},
                    fixed_source_ranges={'en/source':[5,195],'hi/source':[30,160]},
                    source_targets={'en/source':190,'hi/source':130})
                reference=Worker(fixed|{'output_prefix':'reference'},root/'reference').run()
                fast=FastWorker(fixed|{'output_prefix':'fast'},root/'fast').run()
            self.assertEqual(reference['totals'],fast['totals'])
            self.assertEqual(len(reference['shards']),len(fast['shards']))
            for a,b in zip(reference['shards'],fast['shards']):
                for name in ('payload','index'):self.assertEqual(a[name]['sha256'],b[name]['sha256'])

    def test_parallel_payloads_and_metadata_match_reference_with_duplicates_and_heldouts(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);client,plan=fixture(root)
            with patch('production.packing_worker.boto3.Session')as session, \
                 patch('production.fast_packing_worker.boto3.Session')as fast_session, \
                 patch('production.packing_worker.publish',side_effect=publish), \
                 patch('production.fast_packing_worker.publish',side_effect=publish):
                session.return_value.client.return_value=client;fast_session.return_value.client.return_value=client
                reference=Worker(plan|{'output_prefix':'reference','source_targets':{'en/source':220,'hi/source':220}},root/'reference').run()
                fast=FastWorker(plan|{'output_prefix':'fast','source_targets':{'en/source':220,'hi/source':220}},root/'fast').run()
            self.assertEqual(reference['totals'],fast['totals'])
            self.assertEqual(len(reference['shards']),len(fast['shards']))
            for a,b in zip(reference['shards'],fast['shards']):
                for name in ('payload','index'):
                    self.assertEqual(a[name]['sha256'],b[name]['sha256'])
            self.assertTrue(fast['all_curriculum_quotas_satisfied'])

    def test_compact_ledger_suffix_reservation_and_fixed_partition_keep_original_offsets(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);client,plan=fixture(root);plan['source_targets']={'en/source':5}
            # A single long source record gives an unambiguous suffix chain.
            jobs=[json.loads(x)for x in Path(plan['jobs_path']).read_text().splitlines()]
            Path(plan['jobs_path']).write_text(json.dumps(jobs[0])+'\n')
            with patch('production.fast_packing_worker.boto3.Session')as session, \
                 patch('production.fast_packing_worker.publish',side_effect=publish):
                session.return_value.client.return_value=client
                first=FastWorker(plan|{'output_prefix':'first'},root/'first').run()
                spec=first['artifacts']['quota-suffix-reservations.jsonl'];b,k=spec['uri'][5:].split('/',1)
                suffixes=[json.loads(x)for x in client.objects[(b,k)].decode().splitlines()]
                self.assertTrue(suffixes);offset=suffixes[0]['next_document_offset']
                reserve=FastWorker(plan|dict(output_prefix='reserve',allocation_id='dense',reservation_only=True,
                    reservation_reuse_suffixes=True,input_ledger=first['artifacts']['selected.sqlite'],
                    input_suffixes=spec,source_targets={'en/source':2}),root/'reserve').run()
                fixed=FastWorker(plan|dict(output_prefix='fixed',allocation_id='fixed',
                    input_ledger=reserve['artifacts']['selected.sqlite'],
                    fixed_selection=reserve['artifacts']['protected-records.jsonl'],fixed_allocation_id='dense',
                    fixed_source_ranges={'en/source':[1,2]},fixed_total_source_targets={'en/source':2},
                    source_targets={'en/source':1}),root/'fixed').run()
                b,k=fixed['shards'][0]['index']['uri'][5:].split('/',1)
                self.assertEqual(json.loads(client.objects[(b,k)])['segments'][0]['document_offset'],offset+1)
                self.assertEqual(fixed['totals']['content_tokens'],1)


if __name__=='__main__':unittest.main()
