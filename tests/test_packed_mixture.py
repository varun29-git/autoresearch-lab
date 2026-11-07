from collections import Counter
from io import BytesIO
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from production.document_packing import ShardWriter, fragments, file_sha
from production.packed_index import descriptor_offsets
from production.packed_mixture import PackedMixture, PackedSequenceReader, permute


def shards():
    result=[]
    for leaf,stratum,n in [('en/ClimbMix','en/web',400),
        ('hi/Specialized','hi/Code-Concepts',300),('hi/Specialized','hi/STEM-SFT',300),
        ('hi/Specialized','hi/rare-Formal-Logic',1)]:
        digest=hashlib.sha256(stratum.encode()).hexdigest()
        result.append({'leaf':leaf,'stratum':stratum,'sequences':n,
            'payload':{'uri':stratum,'sha256':digest}})
    return result


class PackedMixtureTests(unittest.TestCase):
    def test_shard_order_is_stable_when_payload_hashes_match(self):
        items=shards();extra=items[1]|{'payload':items[1]['payload']|{'uri':'second-code-shard'}}
        items.append(extra)
        a=PackedMixture(items,seed='same-hash');b=PackedMixture(list(reversed(items)),seed='same-hash')
        self.assertEqual([a.locate(i)for i in range(150)], [b.locate(i)for i in range(150)])

    def test_permutation_has_no_duplicates_or_dropped_tail(self):
        for size in list(range(1,130))+[255,256,257,4096,4097]:
            self.assertEqual({permute(i,size,739)for i in range(size)},set(range(size)))

    def test_exact_child_counts_rare_source_and_distributed_resume(self):
        mix=PackedMixture(shards(),seed='frozen',block_sequences=64)
        identities=[(s['payload']['uri'],r)for s,r in(map(mix.locate,range(mix.total_sequences)))]
        self.assertEqual(len(set(identities)),mix.total_sequences)
        self.assertEqual(Counter(s for s,r in identities),Counter({s['stratum']:s['sequences']for s in shards()}))
        distributed=[]
        for rank in range(8):distributed.extend(mix.locate(i)for i in range(rank,mix.total_sequences,8))
        self.assertEqual({(s['payload']['uri'],r)for s,r in distributed},set(identities))
        restored=PackedMixture(list(reversed(shards())),seed='frozen',block_sequences=64)
        self.assertEqual([restored.locate(i)for i in range(500,600)],
                         [mix.locate(i)for i in range(500,600)])
        for start in range(0,960,64):
            children={mix.locate(i)[0]['stratum']for i in range(start,start+64)}
            self.assertIn('hi/Code-Concepts',children);self.assertIn('hi/STEM-SFT',children)
            self.assertIn('en/web',children)

    def test_reader_keeps_document_targets_and_masks_after_mixing(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);catalog=[];paths={}
            for name in ['code','stem']:
                writer=ShardWriter(root,name,length=8)
                for i in range(10):
                    writer.add(list(fragments(np.asarray([100+i,200+i,300+i]),{'id':name+str(i)},length=8)))
                shard=writer.close();ip=Path(shard['index']['path']);op=root/(name+'.offsets')
                with op.open('wb')as out:
                    descriptor_offsets([ip.read_bytes()],out,expected_bytes=ip.stat().st_size,
                        expected_sha256=file_sha(ip),sequences=10)
                shard.update(leaf='hi/Specialized',stratum=name)
                shard['index_offsets']={'path':str(op),'bytes':op.stat().st_size,'sha256':file_sha(op)}
                for field in ['payload','index','index_offsets']:
                    shard[field]['uri']=shard[field]['path'];paths[shard[field]['uri']]=shard[field]['path']
                catalog.append(shard)
            reader=PackedSequenceReader(PackedMixture(catalog,seed='test',block_sequences=8),
                                        resolve_path=lambda uri:paths[uri],max_open_shards=1)
            for i in range(20):
                sample=reader.sample(i);x=sample['inputs'];y=sample['targets']
                self.assertEqual(list(y[:3]),[x[1],x[2],4]);self.assertTrue(np.all(y[3:]==-100))
                self.assertEqual(list(sample['segment_ids'][:4]),[0,0,0,0])
                self.assertTrue(np.all(sample['segment_ids'][4:]==-1))
                self.assertEqual(sample['content_tokens'],3)


if __name__=='__main__':unittest.main()
