import hashlib
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from production.document_packing import file_sha
from production.packing_worker import Exclusions, Worker, fingerprint, integer_split


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        import io
        return {'Body': io.BytesIO(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[(bucket, key)])

    def head_object(self, Bucket, Key, **kwargs):
        raw = self.objects[(Bucket,Key)]
        return {'ContentLength':len(raw),
                'ChecksumSHA256':base64.b64encode(hashlib.sha256(raw).digest()).decode()}


def fake_publish(client, bucket, path, key):
    raw = Path(path).read_bytes()
    client.put_object(Bucket=bucket, Key=key, Body=raw)
    return {'uri': 's3://'+bucket+'/'+key, 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


class PackingWorkerTests(unittest.TestCase):
    def test_invalid_fallback_is_rejected_before_any_io(self):
        from production.fast_packing_worker import FastWorker
        from production.packing_worker import validate_source_fallbacks
        for fallbacks in [{'en/z': 'hi/a'}, {'en/z': 'en/missing'},
                          {'en/z': 'en/z'}, {'en/z': 'en/a', 'en/a': 'en/b'}]:
            plan = {'source_targets': {'en/a': 1, 'en/b': 1, 'en/z': 1, 'hi/a': 1},
                    'source_fallbacks': fallbacks}
            for worker in [Worker, FastWorker]:
                with patch('production.packing_worker.boto3.Session') as session:
                    with self.assertRaises(ValueError):
                        worker(plan, '/unused')
                    session.assert_not_called()
        validate_source_fallbacks({'source_targets': {'en/a': 1, 'en/z': 1},
                                  'source_fallbacks': {'en/z': 'en/a'}})

    def test_backward_source_fallback_uses_fresh_receiver_capacity_and_preserves_language(self):
        from production.preparation_queue import validate_completion
        with tempfile.TemporaryDirectory() as t:
            jobs=Path(t)/'jobs'
            jobs.write_text(''.join(json.dumps({'packing_leaf':leaf,'packing_stratum':leaf,
                'kind':'fresh_native_source'})+'\n' for leaf in ['te/Nemotron-HQ','te/Sangraha']))
            worker=object.__new__(Worker)
            worker.plan={'jobs_path':str(jobs),'source_targets':{'te/Nemotron-HQ':10,'te/Sangraha':10},
                         'source_fallbacks':{'te/Sangraha':'te/Nemotron-HQ'}}
            worker.event=lambda *a,**kw:None
            available={'te/Nemotron-HQ':18,'te/Sangraha':2}
            def process(leaf,stratum,js,target):
                selected=min(target,available[leaf]);available[leaf]-=selected
                return {'leaf':leaf,'stratum':stratum,'target':target,
                        'selected':selected,'shortfall':target-selected}
            worker.process_stratum=process
            results=worker.select_sources()
            self.assertEqual(available,{'te/Nemotron-HQ':0,'te/Sangraha':0})
            self.assertEqual(sum(r['selected'] for r in results),20)
            self.assertTrue(validate_completion(worker.plan,{'physical_creation_finished':True,
                'all_curriculum_quotas_satisfied':True,'totals':{'content_tokens':20},'results':results}))
            available.update({'te/Nemotron-HQ':14,'te/Sangraha':2})
            results=worker.select_sources()
            self.assertEqual(sum(r['selected'] for r in results),16)
            self.assertEqual(sum(r['shortfall'] for r in results),4)
            worker.plan['source_fallbacks']={'te/Sangraha':'te/Nemotron-HQ','te/Nemotron-HQ':'te/Sangraha'}
            with self.assertRaisesRegex(ValueError,'cycle'):worker.select_sources()

    def test_authorized_child_fallback_keeps_source_budget_and_unfilled_deficits(self):
        from production.preparation_queue import validate_completion
        with tempfile.TemporaryDirectory() as t:
            jobs = Path(t)/'jobs.jsonl'
            jobs.write_text(''.join(json.dumps({'packing_leaf':'ta/source','packing_stratum':s,
                'kind':'fresh_native_source'})+'\n' for s in ['rare','native']))
            worker = object.__new__(Worker)
            worker.plan = {'jobs_path':str(jobs),'source_targets':{'ta/source':10},
                'source_stratum_fallbacks':{'ta/source':['native']},
                'stratum_fallback_max_tokens':{'ta/source':3}}
            available = {'rare':2,'native':9}
            worker.event = lambda *args,**kw:None
            def process(leaf,stratum,js,target):
                selected = min(available[stratum],target);available[stratum]-=selected
                return {'leaf':leaf,'stratum':stratum,'target':target,
                        'selected':selected,'shortfall':target-selected}
            worker.process_stratum = process
            results = worker.select_sources()
            self.assertEqual(sum(r['selected'] for r in results),10)
            self.assertEqual(available['native'],1)
            self.assertTrue(validate_completion(worker.plan,{'physical_creation_finished':True,
                'all_curriculum_quotas_satisfied':True,'totals':{'content_tokens':10},'results':results}))
            available.update(rare=2,native=6)
            results = worker.select_sources()
            self.assertEqual(sum(r['selected'] for r in results),8)
            self.assertEqual(sum(r['shortfall'] for r in results),2)
            with self.assertRaises(ValueError):
                validate_completion(worker.plan,{'physical_creation_finished':True,
                    'all_curriculum_quotas_satisfied':False,'totals':{'content_tokens':8},'results':results})

    def test_integer_quotas_conserve_rare_sources(self):
        out = integer_split(101, {'major': 99, 'rare': 1})
        self.assertEqual(out, {'major': 100, 'rare': 1})
        self.assertEqual(sum(integer_split(3, {'a': 1, 'b': 1, 'c': 1}).values()), 3)

    def test_frozen_holdouts_match_translations_and_explicit_families(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'exclusions'
            p.write_text(json.dumps({'family_id': fingerprint(['explicit', 'origin']),
                'locations': [{'doc_id': 'shared-id', 'row_key': 'old:7'}]})+'\n')
            x = Exclusions(p, file_sha(p))
            base = dict(validation=False, doc_id='new-id', row_key='new:8')
            self.assertTrue(x.excludes(base | {'split_family_id': 'origin'}))
            self.assertTrue(x.excludes(base | {'doc_id': 'shared-id'}))
            self.assertTrue(x.excludes(base | {'row_key': 'old:7'}))
            self.assertTrue(x.excludes(base | {'validation': True}))
            self.assertFalse(x.excludes(base | {'split_family_id': 'unrelated'}))
            with self.assertRaises(ValueError):
                Exclusions(p, 'bad-hash')

    def test_end_to_end_excludes_heldout_duplicates_and_preserves_quota_suffix(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            exclusion = root/'exclusions.jsonl'
            exclusion.write_text(json.dumps({'family_id': fingerprint(['published_id', 'held']),
                'locations': [{'doc_id': 'held', 'row_key': 'raw:0'}]})+'\n')
            client = FakeS3()
            rows = []
            contents = []
            for i, (uid, values, text_sha) in enumerate([
                ('held', [900]*5, 'held-sha'),
                ('a', [100]*7, 'duplicate-sha'),
                ('b', [100]*7, 'duplicate-sha'),
                ('c', [200]*31, 'long-sha')]):
                rows.append(dict(doc_id=uid, row_key='raw:'+str(i), split_family_id=uid,
                    validation=False, token_start=len(contents), n_tokens=len(values),
                    source_text_sha256=text_sha))
                contents.extend(values)
            bin_path = root/'input.bin'
            bin_path.write_bytes(np.array(contents, dtype='<u2').tobytes())
            index_path = root/'input.jsonl'
            index_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            binary = fake_publish(client, 'bucket', bin_path, 'bin')
            index = fake_publish(client, 'bucket', index_path, 'index')
            job = dict(job_id='job', source_key='raw', packing_leaf='en/source',
                packing_stratum='en/source', parts=[dict(index_key='old-index',
                    validation=False, token_snapshot={'bytes': len(contents)*2})])
            conversion = dict(status='verified', source_key='raw', job_id='job',
                original_holdout_assignments_preserved=True,
                encoder_identity=dict(tokenizer_sha256='tokenizer', roundtrip_checked_per_document=True),
                outputs=[dict(original_index_key='old-index', validation=False,
                    tokens=len(contents), documents=len(rows), bin=binary, index=index)])
            client.put_object(Bucket='bucket', Key='converted/progress/job.json',
                              Body=json.dumps(conversion).encode())
            jobs = root/'jobs.jsonl'
            jobs.write_text(json.dumps(job)+'\n')
            plan = dict(bucket='bucket', output_prefix='packed', exclusions_path=str(exclusion),
                exclusions_sha256=file_sha(exclusion), deadline_unix=time.time()+600,
                conversion_prefixes=['converted'], tokenizer_sha256='tokenizer',
                plan_sha256='plan', seed='test', sequence_length=8, sequences_per_shard=2,
                packing_buffer_fragments=4, jobs_path=str(jobs), source_targets={'en/source': 20})
            with patch('production.packing_worker.boto3.Session') as session, \
                    patch('production.packing_worker.publish', side_effect=fake_publish):
                session.return_value.client.return_value = client
                worker = Worker(plan, root/'work')
                report = worker.run()
            self.assertEqual(report['totals']['content_tokens'], 20)
            self.assertTrue(report['all_curriculum_quotas_satisfied'])
            self.assertFalse(report['training_manifest_published'])
            self.assertIn(('bucket', 'packed/audit/quota-suffix-reservations.jsonl'), client.objects)
            docs = []
            for (bucket, key), raw in client.objects.items():
                if key.endswith('.segments.jsonl'):
                    for line in raw.splitlines():
                        docs.extend(s['document']['doc_id'] for s in json.loads(line)['segments'])
            self.assertNotIn('held', docs)
            self.assertFalse('a' in docs and 'b' in docs)

    def test_chain_preserves_suffix_and_protects_future_records(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);client=FakeS3();ex=root/'exclusions';ex.write_text('')
            rows=[];values=[]
            for i,n in enumerate((31,29,27)):
                rows.append(dict(doc_id=str(i),row_key='raw:'+str(i),split_family_id=str(i),
                    validation=False,token_start=len(values),n_tokens=n,source_text_sha256='sha-'+str(i)))
                values.extend(range(100+i*100,100+i*100+n))
            bp=root/'bin';bp.write_bytes(np.array(values,dtype='<u2').tobytes())
            ip=root/'idx';ip.write_text(''.join(json.dumps(r)+'\n'for r in rows))
            binary=fake_publish(client,'bucket',bp,'bin');index=fake_publish(client,'bucket',ip,'index')
            job=dict(job_id='job',source_key='raw',packing_leaf='en/source',packing_stratum='en/source',
                     parts=[dict(index_key='old',validation=False,token_snapshot={'bytes':len(values)*2})])
            conversion=dict(status='verified',source_key='raw',job_id='job',
                original_holdout_assignments_preserved=True,
                encoder_identity=dict(tokenizer_sha256='tok',roundtrip_checked_per_document=True),
                outputs=[dict(original_index_key='old',validation=False,tokens=len(values),documents=3,bin=binary,index=index)])
            client.put_object(Bucket='bucket',Key='converted/progress/job.json',Body=json.dumps(conversion).encode())
            jp=root/'jobs';jp.write_text(json.dumps(job)+'\n')
            common=dict(bucket='bucket',exclusions_path=str(ex),exclusions_sha256=file_sha(ex),
                deadline_unix=time.time()+600,conversion_prefixes=['converted'],tokenizer_sha256='tok',
                plan_sha256='plan',seed='test',sequence_length=8,sequences_per_shard=2,
                packing_buffer_fragments=4,jobs_path=str(jp))
            with patch('production.packing_worker.boto3.Session') as session, \
                    patch('production.packing_worker.publish',side_effect=fake_publish):
                session.return_value.client.return_value=client
                first=Worker(common|dict(output_prefix='first',source_targets={'en/source':5}),root/'first').run()
                suffixes=first['artifacts']['quota-suffix-reservations.jsonl']
                prior=json.loads(client.objects[('bucket',suffixes['uri'].split('/',3)[3])])
                reserve=Worker(common|dict(output_prefix='reserve',allocation_id='qat',reservation_only=True,
                    input_ledger=first['artifacts']['selected.sqlite'],input_suffixes=suffixes,
                    source_targets={'en/source':6}),root/'reserve').run()
                self.assertEqual(reserve['totals']['reserved_content_tokens'],6)
                self.assertGreater(reserve['totals']['protected_whole_record_tokens'],6)
                self.assertFalse(reserve['physical_creation_finished'])
                self.assertEqual(reserve['shards'],[])
                fixed=Worker(common|dict(output_prefix='fixed',allocation_id='qat-payload',
                    input_ledger=reserve['artifacts']['selected.sqlite'],
                    fixed_selection=reserve['artifacts']['protected-records.jsonl'],
                    fixed_allocation_id='qat',source_targets={'en/source':6}),root/'fixed').run()
                self.assertEqual(fixed['totals']['content_tokens'],6)
                self.assertTrue(fixed['all_curriculum_quotas_satisfied'])
                slices=[]
                for number,(lo,hi) in enumerate(((0,2),(2,6))):
                    part=Worker(common|dict(output_prefix='slice'+str(number),allocation_id='qat-slice',
                        input_ledger=reserve['artifacts']['selected.sqlite'],
                        fixed_selection=reserve['artifacts']['protected-records.jsonl'],
                        fixed_allocation_id='qat',source_targets={'en/source':hi-lo},
                        fixed_total_source_targets={'en/source':6},
                        fixed_source_ranges={'en/source':[lo,hi]}),root/('slice'+str(number))).run()
                    shard=part['shards'][0];b,k=shard['payload']['uri'][5:].split('/',1)
                    slices.extend(np.frombuffer(client.objects[(b,k)],dtype='<u2')[:hi-lo].tolist())
                    b,k=shard['index']['uri'][5:].split('/',1)
                    self.assertEqual(json.loads(client.objects[(b,k)])['segments'][0]['document_offset'],lo)
                b,k=fixed['shards'][0]['payload']['uri'][5:].split('/',1)
                self.assertEqual(slices,np.frombuffer(client.objects[(b,k)],dtype='<u2')[:6].tolist())
                second=Worker(common|dict(output_prefix='second',allocation_id='dense-01',
                    input_ledger=reserve['artifacts']['selected.sqlite'],
                    input_suffixes=reserve['artifacts']['quota-suffix-reservations.jsonl'],
                    source_targets={'en/source':4}),root/'second').run()
                self.assertEqual(second['totals']['content_tokens'],4)
                spec=second['shards'][0]['index'];b,k=spec['uri'][5:].split('/',1)
                descriptor=json.loads(client.objects[(b,k)])
                segment=descriptor['segments'][0]
                self.assertEqual(segment['document']['source_text_sha256'],prior['source_text_sha256'])
                self.assertEqual(segment['document_offset'],5)
                self.assertFalse(segment['document_end'])
                suffix=second['artifacts']['quota-suffix-reservations.jsonl'];b,k=suffix['uri'][5:].split('/',1)
                self.assertEqual(json.loads(client.objects[(b,k)])['next_document_offset'],9)
                tail=Worker(common|dict(output_prefix='tail-reserve',allocation_id='dense-reserve',
                    reservation_only=True,reservation_reuse_suffixes=True,
                    input_ledger=second['artifacts']['selected.sqlite'],
                    input_suffixes=second['artifacts']['quota-suffix-reservations.jsonl'],
                    source_targets={'en/source':3}),root/'tail-reserve').run()
                tail_payload=Worker(common|dict(output_prefix='tail-payload',allocation_id='dense-payload',
                    input_ledger=tail['artifacts']['selected.sqlite'],
                    fixed_selection=tail['artifacts']['protected-records.jsonl'],
                    fixed_allocation_id='dense-reserve',source_targets={'en/source':3}),root/'tail-payload').run()
                b,k=tail_payload['shards'][0]['index']['uri'][5:].split('/',1)
                self.assertEqual(json.loads(client.objects[(b,k)])['segments'][0]['document_offset'],9)
                tail_next=Worker(common|dict(output_prefix='tail-next',allocation_id='dense-next',
                    input_ledger=tail['artifacts']['selected.sqlite'],
                    input_suffixes=tail['artifacts']['quota-suffix-reservations.jsonl'],
                    source_targets={'en/source':2}),root/'tail-next').run()
                b,k=tail_next['shards'][0]['index']['uri'][5:].split('/',1)
                self.assertEqual(json.loads(client.objects[(b,k)])['segments'][0]['document_offset'],12)
                import sqlite3
                db=sqlite3.connect(root/'second/selected.sqlite')
                self.assertEqual(db.execute("SELECT COUNT(*) FROM selected WHERE allocation='protected:qat'").fetchone()[0],1)
                db.close()
                b,k=reserve['artifacts']['protected-records.jsonl']['uri'][5:].split('/',1)
                protected=json.loads(client.objects[(b,k)])['original_content_tokens']
                rest=Worker(common|dict(output_prefix='rest',allocation_id='rest',
                    input_ledger=second['artifacts']['selected.sqlite'],
                    input_suffixes=second['artifacts']['quota-suffix-reservations.jsonl'],
                    source_targets={'en/source':len(values)-protected-9}),root/'rest').run()
                self.assertTrue(rest['all_curriculum_quotas_satisfied'])
                with patch.object(client,'download_file',wraps=client.download_file) as download:
                    exhausted=Worker(common|dict(output_prefix='exhausted',allocation_id='exhausted',
                        input_ledger=rest['artifacts']['selected.sqlite'],
                        source_targets={'en/source':1}),root/'exhausted').run()
                self.assertEqual(exhausted['results'][0]['rejections']['previously_drained_outputs'],1)
                self.assertEqual(exhausted['results'][0]['selected'],0)
                self.assertEqual(download.call_count,1)  # predecessor ledger only, no repeated source index/bin


if __name__ == '__main__':
    unittest.main()
