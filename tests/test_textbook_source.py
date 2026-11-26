import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from production.document_packing import file_sha
from production.textbook_source import convert, english_eligible
from tests.test_packing_worker import FakeS3, fake_publish


class Classifier:
    def predict(self, text, k):
        return (['__label__zh'], [0.99]) if '中文' in text else (['__label__en'], [0.99])


class Encoder:
    def encode_batch_bytes(self, texts):
        return [np.array([100+ord(c) for c in t], dtype='<u2').tobytes() for t in texts]


class TextbookSourceTests(unittest.TestCase):
    def test_english_prefix_does_not_hide_chinese_suffix(self):
        self.assertTrue(english_eligible('A theorem and its proof.', Classifier()))
        self.assertFalse(english_eligible('中文内容', Classifier()))
        self.assertFalse(english_eligible('An English introduction. '*3000+'中文内容', Classifier()))

    def test_pinned_raw_source_becomes_complete_native_records(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            source = root/'source.parquet'
            text = 'An equation:\n\nx = 4. Preserve the formatting.'
            pq.write_table(pa.table({'uid': ['en-1', 'zh-1'],
                'content': [text, '中文内容']}), source)
            job = dict(job_id='fresh', source_key='textbook/file', source_url='https://example.invalid/file',
                source_sha256=file_sha(source), source_bytes=source.stat().st_size,
                validation_split_scheme='content_sha256_mod10000_v1')
            plan = dict(bucket='bucket', output_prefix='books', deadline_unix=time.time()+600,
                        language_model_sha256='lid-sha')
            client = FakeS3()
            with patch('production.textbook_source.urlopen', return_value=io.BytesIO(source.read_bytes())), \
                    patch('production.textbook_source.publish', side_effect=fake_publish):
                tokens = convert(job, plan, client, root/'work', Classifier(), Encoder(),
                                 {'tokenizer_sha256': 'tokenizer'})
            result = json.loads(client.objects[('bucket','books/progress/fresh.json')])
            self.assertEqual(result['status'], 'verified')
            self.assertEqual(result['counts']['language_rejected_rows'], 1)
            out = next(x for x in result['outputs'] if not x['validation'])
            self.assertEqual(tokens, len(text))
            self.assertEqual(out['documents'], 1)
            b,k = out['index']['uri'][5:].split('/',1)
            record = json.loads(client.objects[(b,k)])
            self.assertEqual(record['doc_id'], 'en-1')
            self.assertEqual(record['n_tokens'], len(text))
            self.assertEqual(record['row_key'], 'textbook/file:0')


if __name__ == '__main__':
    unittest.main()
