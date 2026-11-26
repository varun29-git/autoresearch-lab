import hashlib
import json
import unittest

from production.source_index_repair import repair_index


class SourceIndexRepairTests(unittest.TestCase):
    def setUp(self):
        self.val_uid = next(str(i) for i in range(100000) if
            int.from_bytes(hashlib.blake2b(str(i).encode(),digest_size=8).digest(),'little')%1000==0)
        self.train_uid = next(str(i) for i in range(100000) if
            int.from_bytes(hashlib.blake2b(str(i).encode(),digest_size=8).digest(),'little')%1000!=0)

    def test_split_offsets_and_metadata_are_preserved(self):
        for validation,uid in [(True,self.val_uid),(False,self.train_uid)]:
            with self.subTest(validation=validation):
                records=[{'doc_id':uid,'n_tokens':2,'token_start':11,'source':'kept'},
                         {'doc_id':uid,'n_tokens':3,'token_start':20,'source':'kept'}]
                raw=''.join(json.dumps(r)+'\n' for r in records).encode()
                corrected,stats=repair_index(raw,10,validation=validation)
                rows=[json.loads(line) for line in corrected.splitlines()]
                self.assertEqual([r['token_start'] for r in rows],[0,2])
                self.assertEqual([r['source'] for r in rows],['kept','kept'])
                self.assertEqual(stats['corrected_offsets'],2)
                again,second=repair_index(corrected,10,validation=validation)
                self.assertEqual(again,corrected)
                self.assertEqual(second['corrected_offsets'],0)

    def test_rejects_wrong_split_missing_tokens_and_overlapping_records(self):
        row={'doc_id':self.train_uid,'n_tokens':2,'token_start':0}
        raw=json.dumps(row).encode()
        for size,validation in [(5,False),(6,False),(4,True)]:
            with self.subTest(size=size,validation=validation),self.assertRaises(ValueError):
                repair_index(raw,size,validation=validation)
        overlapping=raw+b'\n'+raw
        with self.assertRaises(ValueError):repair_index(overlapping,8,validation=False)

    def test_empty_validation_bin_and_invalid_metadata(self):
        self.assertEqual(repair_index(b'',0,validation=True)[0],b'')
        for row in [{'doc_id':None,'n_tokens':2,'token_start':0},
                    {'doc_id':self.train_uid,'n_tokens':True,'token_start':0}]:
            with self.assertRaises(ValueError):repair_index(json.dumps(row).encode(),4,validation=False)


if __name__=='__main__':unittest.main()
