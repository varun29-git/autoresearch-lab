import base64
import unittest
from unittest.mock import Mock, patch

from production.preparation_queue import object_spec, ready_tasks, tranches, validate_completion, verify_artifact, recover_reserved_selection
from production.preparation_dispatcher import Dispatcher, resolved_task


class QueueTests(unittest.TestCase):
    def test_saved_reservation_recovery_never_invents_tokens(self):
        task = {'source_targets':{'ta/source':10,'te/source':20},'reservation_only':True}
        report = {'plan_sha256':'old','reservation_creation_finished':True,
                  'physical_creation_finished':False,'shards':[],
                  'all_curriculum_quotas_satisfied':False,'totals':{'reserved_content_tokens':27},
                  'artifacts':{'ledger':'unchanged'},
                  'results':[{'leaf':'ta/source','stratum':'rare','target':10,'selected':9,'shortfall':1},
                             {'leaf':'te/source','stratum':'rare','target':20,'selected':18,'shortfall':2}]}
        approved = {('ta/source','rare'):1,('te/source','rare'):2}
        revised, receipt = recover_reserved_selection(task,report,approved,'reviewed')
        self.assertEqual(revised['source_targets'],{'ta/source':9,'te/source':18})
        self.assertEqual(receipt['totals'],report['totals'])
        self.assertEqual(receipt['artifacts'],report['artifacts'])
        self.assertEqual(receipt['recovery']['deferred_content_tokens'],3)
        self.assertTrue(validate_completion(revised,receipt,'reviewed'))
        self.assertFalse(report['all_curriculum_quotas_satisfied'])
        self.assertEqual(task['source_targets']['ta/source'],10)
        with self.assertRaises(ValueError):
            recover_reserved_selection(task,report,{('ta/source','rare'):3},'reviewed')
        with self.assertRaises(ValueError):
            recover_reserved_selection(task,report|{'totals':{'reserved_content_tokens':30}},approved,'reviewed')

    def test_launch_failure_terminates_cpu_and_cannot_retry(self):
        dispatcher=object.__new__(Dispatcher)
        dispatcher.q={'run':'test','fleet_tag':'test-fleet','allowed_instance_types':['c7a.4xlarge'],
            'ami':'ami-test','worker_profile':'profile','subnet':'subnet-test','security_group':'sg-test',
            'deadline_role_arn':'arn:deadline'}
        dispatcher.bucket='bucket';dispatcher.prefix='queue';dispatcher.state={'tasks':{}}
        dispatcher.s3=Mock();dispatcher.ec2=Mock();dispatcher.scheduler=Mock();dispatcher.save=Mock()
        dispatcher.ec2.run_instances.return_value={'Instances':[{'InstanceId':'i-test'}]}
        dispatcher.scheduler.exceptions.ConflictException=type('ConflictException',(Exception,),{})
        dispatcher.scheduler.create_schedule.side_effect=RuntimeError('deadline service unavailable')
        task={'id':'book','kind':'textbook','runtime_seconds':1800,'jobs_file':'jobs.jsonl',
              'jobs_sha256':'a'*64,'instance_type':'c7a.4xlarge','volume_gb':96,
              'depends_on':[],'plan':{'bucket':'bucket'}}
        with patch('production.preparation_dispatcher.bootstrap',return_value='bootstrap'):
            dispatcher.launch(task)
        record=dispatcher.state['tasks']['book']
        self.assertEqual(record['status'],'failed')
        self.assertFalse(record['automatic_paid_retry'])
        dispatcher.ec2.terminate_instances.assert_called_once_with(InstanceIds=['i-test'])
        self.assertEqual(ready_tasks([task],dispatcher.state['tasks']),[])
        # Launch state and its immutable task hash were persisted before EC2.
        self.assertIn('expected_plan_sha256',record)

        # A lost EC2 response must not leave an untracked paid worker behind.
        dispatcher.state={'tasks':{}}
        dispatcher.ec2.reset_mock()
        dispatcher.ec2.run_instances.side_effect=TimeoutError('lost launch response')
        dispatcher.ec2.describe_instances.return_value={'Reservations':[{'Instances':[{'InstanceId':'i-ambiguous'}]}]}
        with patch('production.preparation_dispatcher.bootstrap',return_value='bootstrap'):
            dispatcher.launch(task)
        self.assertEqual(dispatcher.state['tasks']['book']['status'],'failed')
        dispatcher.ec2.terminate_instances.assert_called_once_with(InstanceIds=['i-ambiguous'])

    def test_resolve_binds_protected_selection_and_predecessor(self):
        task={'id':'bridge','kind':'packing','runtime_seconds':100,'jobs_file':'jobs.jsonl',
              'jobs_sha256':'a'*64,'ledger_from':'reserve','fixed_selection_from':'reserve',
              'source_targets':{'en/STEM':6,'en/book':4},'plan':{'bucket':'b'}}
        artifacts={k:{'uri':'s3://b/'+k,'sha256':'b'*64,'bytes':10} for k in
                   ['selected.sqlite','quota-suffix-reservations.jsonl','protected-records.jsonl']}
        state={'reserve':{'status':'complete','artifacts':artifacts,
                          'effective_source_targets':{'en/STEM':3,'en/book':7}}}
        result=resolved_task(task,state,'b',1000)
        self.assertEqual(result['plan']['deadline_unix'],1100)
        self.assertEqual(result['plan']['fixed_selection'],artifacts['protected-records.jsonl'])
        self.assertEqual(result['source_targets'],state['reserve']['effective_source_targets'])
        self.assertNotIn('input_ledger',task['plan'])
        first=resolved_task(task|{'fixed_partition_sizes':[5,5],'fixed_partition_index':0},state,'b',1000)
        last=resolved_task(task|{'fixed_partition_sizes':[5,5],'fixed_partition_index':1},state,'b',1000)
        for k in state['reserve']['effective_source_targets']:
            self.assertEqual(first['plan']['fixed_source_ranges'][k][1],last['plan']['fixed_source_ranges'][k][0])
            self.assertEqual(last['plan']['fixed_source_ranges'][k][1],state['reserve']['effective_source_targets'][k])
        self.assertEqual(sum(first['source_targets'].values()),5)
        self.assertEqual(sum(last['source_targets'].values()),5)
        state['reserve']['status']='failed'
        with self.assertRaises(ValueError):
            resolved_task(task,state,'b',1000)

    def test_rounding_preserves_each_source_and_language(self):
        quotas = {'en/a': 13, 'en/b': 7, 'hi/a': 11, 'ta/a': 9}
        parts = tranches(quotas, [10, 10, 10, 10])
        self.assertEqual([sum(p.values()) for p in parts], [10]*4)
        self.assertEqual([sum(v for k,v in p.items() if k.startswith('en/')) for p in parts], [5]*4)
        self.assertEqual({k:sum(p.get(k,0) for p in parts) for k in quotas}, quotas)
        with self.assertRaises(ValueError):
            tranches(quotas, [39])

    def test_only_complete_dependencies_release_work(self):
        tasks = [{'id':'a','depends_on':['warmup']}, {'id':'b','depends_on':['a']}]
        self.assertEqual(ready_tasks(tasks, {'warmup':{'status':'failed'}}), [])
        self.assertEqual([t['id'] for t in ready_tasks(tasks, {'warmup':{'status':'complete'}})], ['a'])
        self.assertEqual(ready_tasks(tasks, {'warmup':{'status':'complete'},'a':{'status':'running'}}), [])

    def test_completion_requires_exact_accounting(self):
        task = {'source_targets':{'en/STEM':6,'en/book':4},'source_fallbacks':{'en/STEM':'en/book'}}
        report = {'all_curriculum_quotas_satisfied':True,'physical_creation_finished':True,
                  'plan_sha256':'frozen','totals':{'content_tokens':10},
                  'results':[{'leaf':'en/STEM','selected':3,'target':3,'shortfall':0},
                             {'leaf':'en/book','selected':7,'target':7,'shortfall':0}]}
        self.assertTrue(validate_completion(task,report,'frozen'))
        with self.assertRaises(ValueError):
            validate_completion(task,report,'different-plan')
        with self.assertRaises(ValueError):
            validate_completion({'source_targets':task['source_targets']},report)
        report['all_curriculum_quotas_satisfied'] = False
        with self.assertRaises(ValueError):
            validate_completion(task,report)

    def test_reservation_is_not_a_physical_completion(self):
        task = {'source_targets':{'en/a':10},'reservation_only':True}
        report = {'all_curriculum_quotas_satisfied':True,'reservation_creation_finished':True,
                  'totals':{'reserved_content_tokens':10},
                  'results':[{'leaf':'en/a','selected':10,'target':10,'shortfall':0}]}
        self.assertTrue(validate_completion(task,report))
        with self.assertRaises(ValueError):
            validate_completion(task|{'reservation_only':False},report)

    def test_full_sha_required_for_multipart_dependencies(self):
        class S3:
            def head_object(self, **kw):
                return self.head
        s3=S3();s3.head={'ContentLength':7,'ChecksumSHA256':base64.b64encode(bytes.fromhex('a'*64)).decode()}
        self.assertEqual(object_spec(s3,'b','k')['sha256'],'a'*64)
        s3.head['ChecksumSHA256']+='-4'
        with self.assertRaises(ValueError):
            object_spec(s3,'b','k')
        s3.head['Metadata']={'sha256':'b'*64}
        self.assertEqual(object_spec(s3,'b','k')['sha256'],'b'*64)
        spec=object_spec(s3,'b','k')|{'s3_checksum_type':'COMPOSITE','multipart_parts':4}
        self.assertTrue(verify_artifact(s3,spec))
        with self.assertRaises(ValueError):
            verify_artifact(s3,spec|{'sha256':'c'*64})


if __name__ == '__main__':
    unittest.main()
