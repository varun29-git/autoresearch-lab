import hashlib
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


def fake_publish(client, bucket, path, key):
    raw = Path(path).read_bytes()
    client.put_object(Bucket=bucket, Key=key, Body=raw)
    return {'uri': 's3://'+bucket+'/'+key, 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


class PackingWorkerTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
