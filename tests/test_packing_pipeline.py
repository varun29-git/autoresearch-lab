from pathlib import Path
import gzip
import hashlib
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from production.fast_packing_worker import FastWorker
from production.parallel_gzip import ParallelGzipWriter
from production.readonly_ledger import ReadOnlyLedger
from tests.test_fast_packing_worker import fixture,publish
from tests.test_fixed_translation_pairs import fixture as paired_fixture
from tests.test_translation_pairs import Encoder

OPTIONS={'read_only_fixed_ledger':True,'parallel_fragment_preparation':True,'overlap_fixed_strata':True}


def hashes(report):
    return [tuple(s[n]['sha256']for n in ('payload','index','index_offsets'))for s in report['shards']]


class PipelineTests(unittest.TestCase):
    def test_readonly_parallel_fixed_ranges_preserve_every_payload_and_boundary(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);client,plan=fixture(root)
            plan|={'source_targets':{'en/source':220,'hi/source':220},'allocation_id':'reserved'}
            with patch('production.fast_packing_worker.boto3.Session')as session,\
                 patch('production.fast_packing_worker.publish',side_effect=publish):
                session.return_value.client.return_value=client
                reserve=FastWorker(plan|{'output_prefix':'reserve','reservation_only':True},root/'reserve').run()
                fixed=plan|{'fixed_selection':reserve['artifacts']['protected-records.jsonl'],
                    'input_ledger':reserve['artifacts']['selected.sqlite'],'fixed_allocation_id':'reserved',
                    'fixed_total_source_targets':plan['source_targets'],
                    'fixed_source_ranges':{'en/source':[5,210],'hi/source':[7,200]},
                    'source_targets':{'en/source':205,'hi/source':193}}
                a=FastWorker(fixed|{'output_prefix':'baseline'},root/'baseline').run()
                worker=FastWorker(fixed|OPTIONS|{'output_prefix':'optimized'},root/'optimized');b=worker.run()
            self.assertEqual(hashes(a),hashes(b));self.assertEqual(a['totals'],b['totals'])
            self.assertEqual(a['results'],b['results'])
            self.assertEqual(b['artifacts']['selected.sqlite'],fixed['input_ledger'])
            self.assertFalse((root/'optimized/selected.sqlite').exists())
            self.assertEqual(worker.ram_used,0);self.assertFalse(worker.pending_release)
            self.assertEqual(sum(worker.path_holds.values()),0)
            ledger=ReadOnlyLedger(root/'optimized/input-ledger.sqlite',cache_mib=1)
            with self.assertRaisesRegex(ValueError,'cannot mutate'):ledger.commit()
            with self.assertRaises(sqlite3.OperationalError):ledger.conn.execute('DELETE FROM allocations')
            ledger.close()

    def test_parallel_pairing_keeps_compound_segments_and_source_charges(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);client,plan,records=paired_fixture(root)
            with patch('production.fast_packing_worker.boto3.Session')as session,\
                 patch('production.fast_packing_worker.publish',side_effect=publish),\
                 patch('terminus.tokenizer_native.Native65kTokenizer',return_value=Encoder()):
                session.return_value.client.return_value=client
                a=FastWorker(plan|{'output_prefix':'baseline'},root/'baseline').run()
                b=FastWorker(plan|OPTIONS|{'output_prefix':'optimized'},root/'optimized').run()
            self.assertEqual(hashes(a),hashes(b));self.assertEqual(a['totals'],b['totals'])
            self.assertEqual(a['translation_pairs'],b['translation_pairs'])

    def test_bulk_reservation_and_compression_keep_source_order_and_ownership(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);client,plan=fixture(root)
            plan|={'source_targets':{'en/source':220,'hi/source':220},'allocation_id':'reserved',
                'reservation_only':True,'compress_reservations':True}
            with patch('production.fast_packing_worker.boto3.Session')as session,\
                 patch('production.fast_packing_worker.publish',side_effect=publish):
                session.return_value.client.return_value=client
                a=FastWorker(plan|{'output_prefix':'baseline'},root/'baseline').run()
                b=FastWorker(plan|{'output_prefix':'optimized','parallel_reservation_compression':True,
                    'ledger_bulk_insert_rows':200,'ledger_commit_rows':1000000},root/'optimized').run()
            self.assertEqual(a['totals'],b['totals']);self.assertEqual(a['results'],b['results'])
            self.assertEqual(gzip.decompress((root/'baseline/protected-records.jsonl.gz').read_bytes()),
                gzip.decompress((root/'optimized/protected-records.jsonl.gz').read_bytes()))
            connections=[sqlite3.connect(root/n/'selected.sqlite')for n in ['baseline','optimized']]
            try:self.assertEqual(*[c.execute('SELECT * FROM selected ORDER BY text_sha').fetchall()for c in connections])
            finally:
                for c in connections:c.close()

    def test_parallel_gzip_is_readable_by_python_and_pair_discovery(self):
        import duckdb
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'records.jsonl.gz';writer=ParallelGzipWriter(path,workers=3,block_bytes=2048)
            expected=b''.join((json.dumps({'number':i,'text':'मराठी తెలుగు café'} ,ensure_ascii=False)+'\n').encode()for i in range(10000))
            for start in range(0,len(expected),997):writer.write(expected[start:start+997])
            writer.close();self.assertEqual(gzip.decompress(path.read_bytes()),expected)
            conn=duckdb.connect()
            try:self.assertEqual(conn.execute('SELECT count(*),sum(number) FROM read_json_auto(?)',[str(path)]).fetchone(),(10000,49995000))
            finally:conn.close()

    def test_corrupt_source_ids_in_child_preparation_cannot_complete(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);client,plan=fixture(root)
            plan|={'source_targets':{'en/source':220},'allocation_id':'reserved'}
            with patch('production.fast_packing_worker.boto3.Session')as session,\
                 patch('production.fast_packing_worker.publish',side_effect=publish):
                session.return_value.client.return_value=client
                reserve=FastWorker(plan|{'output_prefix':'reserve','reservation_only':True},root/'reserve').run()
                # Simulate a matching artifact receipt containing an illegal reserved token:
                # digest verification passes, but child token validation must fail.
                data=bytearray(client.objects[('bucket','input/bin0')]);data[:]=b'\x04\x00'*(len(data)//2)
                client.objects[('bucket','input/bin0')]=bytes(data)
                spec=reserve['artifacts']['protected-records.jsonl'];k=spec['uri'][len('s3://bucket/'):]
                rows=[json.loads(l)for l in client.objects[('bucket',k)].splitlines()]
                for row in rows:row['source_bin_sha256']=hashlib.sha256(data).hexdigest()
                content=b''.join((json.dumps(row)+'\n').encode()for row in rows);client.objects[('bucket',k)]=content
                fixed=plan|OPTIONS|{'output_prefix':'bad','fixed_selection':spec|{'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()},
                    'input_ledger':reserve['artifacts']['selected.sqlite'],'fixed_allocation_id':'reserved'}
                with self.assertRaisesRegex(ValueError,'reserved IDs'):FastWorker(fixed,root/'bad').run()
            self.assertNotIn(('bucket','bad/MATERIALIZATION.json'),client.objects)


if __name__=='__main__':unittest.main()
