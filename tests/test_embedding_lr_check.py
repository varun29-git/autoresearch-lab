import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from production.embedding_lr_check import ARMS, config_for, run, stage, validate
from production.h100_24h import campaign
from terminus.integrity import file_sha256, fingerprint


def manifests(root):
    paths = []
    for name, tokens, microbatch in [('proxy', campaign.PROXY_SCHEDULE['total_tokens'], 8),
                                     ('full', campaign.CONFIRMATION_SCHEDULE['total_tokens'], 2)]:
        p = Path(root) / f'{name}-input.json'
        p.write_text(json.dumps({'config': {'seq_len': 4096, 'batch_size': microbatch,
                                           'total_train_tokens': tokens},
                                 'shards': [], 'tokenizer': {}, 'provenance': {}}))
        paths.append(p)
    return paths


def packet(root):
    a, b = manifests(root)
    out = Path(root) / 'check'
    stage(proxy_manifest=a, full_manifest=b, output_dir=out)
    return out


class EmbeddingLRCheckTests(unittest.TestCase):
    def test_transfer_and_only_embedding_changes(self):
        for geometry, microbatch, rates in [('proxy', 8, (0.006, 0.003)),
                                            ('full_size', 2, (0.002, 0.001))]:
            a = config_for(geometry, 'reference', microbatch=microbatch)
            b = config_for(geometry, 'challenger', microbatch=microbatch)
            self.assertEqual((a['optimizer']['adamw_lr'], b['optimizer']['adamw_lr']), rates)
            self.assertEqual(b['optimizer']['muon_lr'], 0.017)
            self.assertEqual(b['optimizer']['norm_lr'], 0.0003)
            self.assertEqual(b['optimizer']['qk_norm_lr'], 0.0001)
            self.assertEqual(b['training']['seed'], a['training']['seed'])
            b['optimizer']['adamw_lr'] = a['optimizer']['adamw_lr']
            self.assertEqual(a, b)

    def test_staging_validates_counts_batch_and_both_shared_manifests(self):
        with tempfile.TemporaryDirectory() as root:
            out = packet(root)
            r = validate(out)
            self.assertEqual(r['geometries']['proxy']['parameters'], 194_329_390)
            self.assertEqual(r['geometries']['full_size']['parameters'], 1_446_478_126)
            for geometry in r['geometries'].values():
                self.assertEqual(geometry['global_batch_tokens'], 2_097_152)
                self.assertEqual(geometry['world_size'], 1)
            self.assertFalse(r['native_run_performed'])
            self.assertFalse(r['production_lr_changed'])

    def test_tampered_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            out = packet(root)
            p = out / 'proxy/challenger.json'
            c = json.loads(p.read_text())
            c['optimizer']['muon_lr'] = 0.02
            p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'file changed'):
                validate(out)

    def test_resigning_packet_cannot_expand_the_allowed_lr_or_model(self):
        with tempfile.TemporaryDirectory() as root:
            out = packet(root)
            p = out / 'proxy/challenger.json'
            c = json.loads(p.read_text())
            c['optimizer']['adamw_lr'] = 0.004
            p.write_text(json.dumps(c))
            identity = out / 'packet.json'
            r = json.loads(identity.read_text())
            r['files']['proxy/challenger.json'] = file_sha256(p)
            r.pop('packet_sha256')
            r['packet_sha256'] = fingerprint(r)
            identity.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError, 'isolated optimizer'):
                validate(out)

    def test_invalid_microbatch_horizon_and_arm_are_rejected(self):
        with self.assertRaises(ValueError):
            config_for('proxy', 'reference', microbatch=3)
        with self.assertRaises(ValueError):
            config_for('proxy', 'unreviewed', microbatch=8)
        with tempfile.TemporaryDirectory() as root:
            a, b = manifests(root)
            payload = json.loads(a.read_text())
            payload['config']['total_train_tokens'] += 2_097_152
            a.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'context/horizon'):
                stage(proxy_manifest=a, full_manifest=b, output_dir=Path(root) / 'wrong')

    def test_production_default_and_validator_remain_locked(self):
        from production.amd.recipe import SELECTED_DENSE_OPTIMIZER, model_for, validate_production
        from terminus.optim import OptimizerConfig
        from terminus.train import TrainConfig
        self.assertEqual(SELECTED_DENSE_OPTIMIZER['adamw_lr'], 0.002)
        with self.assertRaisesRegex(ValueError, 'reviewed AMD'):
            validate_production(model_for(), TrainConfig(device='cuda', dtype='bf16'),
                                OptimizerConfig(**(SELECTED_DENSE_OPTIMIZER | {'adamw_lr': 0.001})))

    def test_native_entry_refuses_cpu_without_starting_training(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(packet=str(packet(root)))
            with patch('torch.cuda.is_available', return_value=False), \
                 patch('terminus.train.prepare_training') as training:
                with self.assertRaisesRegex(ValueError, 'human-launched H100'):
                    run(args)
                training.assert_not_called()


if __name__ == '__main__':
    unittest.main()
