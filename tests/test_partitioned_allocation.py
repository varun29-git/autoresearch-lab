import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from production.document_packing import file_sha
from production.packing_worker import fingerprint
from production.partitioned_allocation import (PartitionAllocator, capacity_split,
    catalogue_record, compile_catalogues, partition_for, phase_targets, reconcile_phase)

try:
    import duckdb
except ImportError:
    duckdb=None


class Exclusions:
    def excludes(self,r):return r.get('validation')is not False or r['split_family_id']=='heldout'


def row(i,n):
    return {'doc_id':str(i),'row_key':'file:'+str(i),'split_family_id':'family-'+str(i),
        'validation':False,'source_text_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
        'source_bin_uri':'s3://bucket/file','source_bin_sha256':'aa'*32,'source_index_sha256':'bb'*32,
        'n_tokens':n,'token_start':i*1000}


class CoordinatorTests(unittest.TestCase):
    def test_exact_capacity_aware_split_handles_scarcity_and_zero_partitions(self):
        for caps in ({0:0,1:1,2:99,3:1000,4:2},{0:0,1:0},{0:17,1:3}):
            for total in range(sum(caps.values())+1):
                split=capacity_split(total,caps)
                self.assertEqual(sum(split.values()),total)
                self.assertTrue(all(split[k]<=caps[k]for k in caps))
        with self.assertRaises(ValueError):capacity_split(4,{0:3})
        with self.assertRaises(ValueError):capacity_split(1,{0:-1})

    def test_catalogue_excludes_prior_and_frozen_holdout(self):
        r=row(1,5)
        self.assertIsNone(catalogue_record(r,leaf='hi/source',stratum='code',source_order='a',
            exclusions=Exclusions(),previously_allocated=True))
        r['split_family_id']='heldout'
        self.assertIsNone(catalogue_record(r,leaf='hi/source',stratum='code',source_order='a',exclusions=Exclusions()))
        r=row(1,5);r['n_tokens']=-1
        with self.assertRaises(ValueError):
            catalogue_record(r,leaf='hi/source',stratum='code',source_order='a',exclusions=Exclusions())

    def test_duplicate_text_always_owns_same_partition(self):
        sha=row(1,10)['source_text_sha256']
        self.assertEqual(partition_for(sha,5),partition_for(sha,5))
        with self.assertRaises(ValueError):partition_for('bad',5)


@unittest.skipUnless(duckdb,'DuckDB 1.4.2 required for native allocation tests')
class NativePartitionTests(unittest.TestCase):
    def test_global_catalogue_canonicalizes_cross_source_duplicates_and_prior(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);rows=[]
            for i,leaf in [(1,'en/preferred'),(1,'en/fallback'),(2,'hi/source'),(3,'en/preferred')]:
                rows.append(catalogue_record(row(i,10),leaf=leaf,stratum='code',source_order='a',exclusions=Exclusions()))
            path=root/'raw.parquet';pq.write_table(pa.Table.from_pylist(rows),path)
            prior=root/'prior.parquet';pq.write_table(pa.Table.from_pylist([{'sha':row(3,10)['source_text_sha256']}]),prior)
            spec=lambda p:{'path':str(p),'bytes':p.stat().st_size,'sha256':file_sha(p)}
            report=compile_catalogues([spec(path)],[spec(prior)],root/'compiled',
                source_priority=['en/preferred','hi/source','en/fallback'],partitions=5,
                predecessor_sha256='cc'*32,exclusions_sha256='dd'*32,threads=1)
            self.assertEqual(report['eligible_unique_records'],2)
            final=[]
            for p in report['catalogues']:
                values=pq.read_table(p['path']).to_pylist();final.extend(values)
                self.assertTrue(all(partition_for(r['sha'],5)==p['partition']for r in values))
            self.assertEqual({r['leaf']for r in final},{'en/preferred','hi/source'})
            incompatible=dict(rows[0],n=99);pq.write_table(pa.Table.from_pylist(rows+[incompatible]),path)
            with self.assertRaisesRegex(ValueError,'incompatible tokenizer lengths'):
                compile_catalogues([spec(path)],[spec(prior)],root/'bad',
                    source_priority=['en/preferred','hi/source','en/fallback'],partitions=5,
                    predecessor_sha256='cc'*32,exclusions_sha256='dd'*32,threads=1)

    def make_engines(self,root,partitions=5):
        random_source=random.Random(22);rows={p:[]for p in range(partitions)}
        for i in range(600):
            r=row(i,random_source.randrange(3,101));leaf='hi/source'if i%2 else'en/source'
            c=catalogue_record(r,leaf=leaf,stratum='code'if i%3 else'math',
                source_order=fingerprint(['seed',r['row_key']]),exclusions=Exclusions())
            rows[partition_for(r['source_text_sha256'],partitions)].append(c)
        engines=[]
        for p in range(partitions):
            path=root/f'part-{p}.parquet';pq.write_table(pa.Table.from_pylist(rows[p]),path)
            spec={'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path)}
            engines.append(PartitionAllocator(root/f'part-{p}.duckdb',[spec],partition=p,
                partitions=partitions,predecessor_sha256='cc'*32,exclusions_sha256='dd'*32,threads=1))
        return engines

    def test_parallel_quotas_exact_and_suffixes_neither_replay_nor_skip(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);engines=self.make_engines(root)
            try:
                caps={p:e.capacities()for p,e in enumerate(engines)}
                totals={k:sum(c.get(k,0)for c in caps.values())for k in caps[0]}
                quotas={k:n//3 for k,n in totals.items()}
                targets=phase_targets(quotas,caps)
                receipts=[e.reserve('dense-01',targets[p],root/f'dense-01-{p}.jsonl',reuse_suffixes=True)
                    for p,e in enumerate(engines)]
                r=reconcile_phase(receipts,phase='dense-01',targets=quotas,partitions=5,
                    exclusions_sha256='dd'*32,predecessor_sha256={p:'cc'*32 for p in range(5)})
                self.assertEqual(r['content_tokens'],sum(quotas.values()))
                first=[json.loads(x)for spec in receipts for x in Path(spec['selection']['path']).read_text().splitlines()]
                self.assertEqual(sum(x['allocated_content_tokens']for x in first),sum(quotas.values()))
                after={p:e.capacities()for p,e in enumerate(engines)}
                self.assertEqual(sum(sum(c.values())for c in after.values()),sum(totals.values())-sum(quotas.values()))
                remainder={k:totals[k]-quotas[k]for k in totals}
                target2=phase_targets(remainder,after)
                receipts2=[e.reserve('dense-02',target2[p],root/f'dense-02-{p}.jsonl',reuse_suffixes=True)
                    for p,e in enumerate(engines)]
                reconcile_phase(receipts2,phase='dense-02',targets=remainder,partitions=5,
                    exclusions_sha256='dd'*32,predecessor_sha256={p:fingerprint(receipts[p])for p in range(5)})
                spans={}
                for spec in receipts+receipts2:
                    for raw in Path(spec['selection']['path']).read_text().splitlines():
                        record=json.loads(raw);sha=record['source_text_sha256']
                        self.assertEqual(partition_for(sha,5),spec['partition'])
                        spans.setdefault(sha,[]).append((record['allocated_document_offset'],record['allocated_content_tokens'],record['original_content_tokens']))
                for parts in spans.values():
                    cursor=0
                    for offset,n,total in sorted(parts):self.assertEqual(offset,cursor);cursor+=n
                    self.assertEqual(cursor,total)
                self.assertFalse(any(e.capacities()for e in engines))
                corrupted=copy.deepcopy(receipts);corrupted[0]['source_targets'][0]['tokens']+=1
                with self.assertRaises(ValueError):
                    reconcile_phase(corrupted,phase='dense-01',targets=quotas,partitions=5,
                        exclusions_sha256='dd'*32,predecessor_sha256={p:'cc'*32 for p in range(5)})
                with self.assertRaises(ValueError):engines[0].reserve('dense-01',{},root/'replay.jsonl')
            finally:
                for e in engines:e.close()

    def test_protected_phase_burns_unused_record_tails(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);engines=self.make_engines(root,1);e=engines[0]
            try:
                key=next(iter(e.capacities()));before=sum(e.capacities().values())
                receipt=e.reserve('qat-finishing',{key:1},root/'protected.jsonl')
                record=json.loads(Path(receipt['selection']['path']).read_text())
                self.assertEqual(before-sum(e.capacities().values()),record['original_content_tokens'])
                self.assertGreater(record['original_content_tokens'],1)
                with self.assertRaises(ValueError):e.reserve('oversize',{key:10**12},root/'bad.jsonl')
                self.assertFalse((root/'bad.jsonl').exists())
            finally:e.close()

    def test_foreign_partition_and_duplicate_source_sha_are_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);r=row(7,10);c=catalogue_record(r,leaf='hi/source',stratum='code',source_order='a',exclusions=Exclusions())
            path=root/'wrong.parquet';pq.write_table(pa.Table.from_pylist([c]),path)
            spec={'path':str(path),'bytes':path.stat().st_size,'sha256':file_sha(path)}
            wrong=(partition_for(r['source_text_sha256'],5)+1)%5
            with self.assertRaises(ValueError):PartitionAllocator(root/'wrong.db',[spec],partition=wrong,partitions=5,
                predecessor_sha256='cc'*32,exclusions_sha256='dd'*32)
            pq.write_table(pa.Table.from_pylist([c,c]),path);spec.update(bytes=path.stat().st_size,sha256=file_sha(path))
            with self.assertRaises(Exception):PartitionAllocator(root/'duplicates.db',[spec],partition=0,partitions=1,
                predecessor_sha256='cc'*32,exclusions_sha256='dd'*32)


if __name__=='__main__':unittest.main()
