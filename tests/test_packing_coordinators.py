from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from botocore.exceptions import ClientError

from production.document_packing import file_sha
from production.fast_packing_worker import FastWorker
from production.packing_coordinators import census,groups_for
from tests.test_fast_packing_worker import fixture,publish
from tests.test_fixed_translation_pairs import fixture as paired_fixture
from tests.test_packing_pipeline import OPTIONS,hashes
from tests.test_translation_pairs import Encoder


def child_in_process(plan_path,work,cancel):
    plan=json.loads(plan_path.read_text());plan['plan_sha256']=file_sha(plan_path)
    return FastWorker(plan,work).run()


class CoordinatorTests(unittest.TestCase):
    def test_census_enforces_exact_quotas_pair_exclusion_and_order(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'fixed'
            rows=[{'leaf':leaf,'allocated_content_tokens':n,'source_text_sha256':sha}
                for leaf,n,sha in [('en',3,'a'),('en',4,'b'),('hi',5,'c')]]
            def save():path.write_text(''.join(json.dumps(r)+'\n'for r in rows))
            save();order,net,records=census(path,{'en':7,'hi':5,'zero':0},{'a'},Counter(en=3))
            self.assertEqual(order,['en','hi']);self.assertEqual(net,{'en':4,'hi':5})
            groups=groups_for(net,records,4);self.assertEqual(len(groups),2)
            self.assertEqual({k:v for g in groups for k,v in g.items()},net)
            with self.assertRaisesRegex(ValueError,'quotas'):census(path,{'en':8,'hi':5},set(),Counter())
            with self.assertRaisesRegex(ValueError,'source charges'):census(path,{'en':7,'hi':5},{'a'},Counter(en=4))
            rows[1]['source_text_sha256']='a';save()
            with self.assertRaisesRegex(ValueError,'repeats'):census(path,{'en':7,'hi':5},{'a'},Counter(en=7))
            rows=[rows[0],rows[2],rows[1]];save()
            self.assertIsNone(census(path,{'en':7,'hi':5},set(),Counter()))

    def check_equivalence(self,paired=False):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            if paired:client,plan,_=paired_fixture(root)
            else:client,plan=fixture(root)
            original_get=client.get_object
            def get(**kwargs):
                try:return original_get(**kwargs)
                except KeyError as exc:
                    raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')from exc
            client.get_object=get
            with patch('production.fast_packing_worker.boto3.Session')as session,\
                 patch('production.fast_packing_worker.publish',side_effect=publish),\
                 patch('production.packing_coordinators.publish',side_effect=publish),\
                 patch('production.packing_coordinators.run_child',side_effect=child_in_process),\
                 patch('terminus.tokenizer_native.Native65kTokenizer',return_value=Encoder()):
                session.return_value.client.return_value=client
                if not paired:
                    plan|={'source_targets':{'en/source':220,'hi/source':220},'allocation_id':'reserved'}
                    reserve=FastWorker(plan|{'output_prefix':'reserve','reservation_only':True},root/'reserve').run()
                    plan|={'fixed_selection':reserve['artifacts']['protected-records.jsonl'],
                        'input_ledger':reserve['artifacts']['selected.sqlite'],'fixed_allocation_id':'reserved',
                        'fixed_total_source_targets':plan['source_targets']}
                plan|=OPTIONS
                a=FastWorker(plan|{'output_prefix':'baseline'},root/'baseline').run()
                b=FastWorker(plan|{'output_prefix':'pooled','materializer_processes':4},root/'pooled').run()
            self.assertEqual(hashes(a),hashes(b));self.assertEqual(a['totals'],b['totals'])
            self.assertEqual(a['results'],b['results']);self.assertEqual(b['coordinators']['workers'],2)
            self.assertEqual(a.get('translation_pairs'),b.get('translation_pairs'))
            self.assertEqual(b['artifacts']['selected.sqlite'],plan['input_ledger'])

    def test_ordinary_outputs_and_quota_suffixes_match(self):self.check_equivalence()
    def test_paired_outputs_and_source_charges_match(self):self.check_equivalence(paired=True)


if __name__=='__main__':unittest.main()
