"""CPU regression tests for horizon views and the token-accounting gate."""

import json
from pathlib import Path
import tempfile
import unittest

from terminus.data import TokenDataConfig, TokenManifest, TokenShardSpec

from production.kimi_h100_5b import campaign
from production.kimi_h100_5b.horizons import (audit_token_accounting, build_horizon_view,
                                              main as horizons_main)


def source_manifest(**config_changes):
    options = {"source_mix": {"en": 0.5, "code": 0.5}}
    options.update(config_changes)
    config = TokenDataConfig(seq_len=4096, batch_size=8,
                             total_train_tokens=6_000_000_000, **options)
    shards = [TokenShardSpec(f"{name}.bin", name, "train", 4_000_000_000, language="en")
              for name in ("en", "code")] + [
              TokenShardSpec("val.bin", "en", "val", 60_000_000, language="en")]
    return TokenManifest(config, shards, {"path": "tokenizer.json", "sha256": "ab" * 32},
                         {"normalization": "NFC", "packing": "continuous_eos",
                          "document_dedup": "sha256_nfc", "licenses": ["cc"]})


class HorizonViewTests(unittest.TestCase):
    def test_view_preserves_data_identity_and_records_derivation(self):
        source = source_manifest()
        view = build_horizon_view(source, total_train_tokens=4_999_610_368, microbatch=4)
        self.assertEqual(view.config.total_train_tokens, 4_999_610_368)
        self.assertEqual(view.config.batch_size, 4)
        self.assertEqual(view.shards, source.shards)
        self.assertEqual(view.tokenizer, source.tokenizer)
        self.assertEqual(view.config.source_mix, {"en": 0.5, "code": 0.5})
        record = view.provenance["horizon_view"]
        self.assertEqual(record["total_train_tokens"], 4_999_610_368)
        self.assertEqual(record["accumulation_steps_at_world_size_1"], 128)
        self.assertTrue(record["shards_unchanged"])
        for key in ("normalization", "packing", "document_dedup", "licenses"):
            self.assertEqual(view.provenance[key], source.provenance[key])

    def test_view_rejects_misaligned_or_uncovered_horizons(self):
        source = source_manifest()
        with self.assertRaisesRegex(ValueError, "complete sequences"):
            build_horizon_view(source, total_train_tokens=4_999_610_367, microbatch=4)
        with self.assertRaisesRegex(ValueError, "usable"):
            build_horizon_view(source, total_train_tokens=6_000_005_120, microbatch=4)
        with self.assertRaisesRegex(ValueError, "divide"):
            build_horizon_view(source, total_train_tokens=4_999_610_368, microbatch=3)
        small = source_manifest()
        small.shards[0] = TokenShardSpec("en.bin", "en", "train", 10_000_000, language="en")
        small.shards[1] = TokenShardSpec("code.bin", "code", "train", 10_000_000, language="en")
        with self.assertRaisesRegex(ValueError, "shards do not cover"):
            build_horizon_view(small, total_train_tokens=4_999_610_368, microbatch=4)
        with self.assertRaisesRegex(ValueError, "prefix"):
            build_horizon_view(source_manifest(cosmopedia_stories_prefix_tokens=5_999_000_000),
                               total_train_tokens=4_999_610_368, microbatch=4)

    def test_phase_curriculum_truncated_consistently_or_rejected(self):
        from production.kimi_h100_5b.horizons import _truncate_phases

        phases = [{"tokens": 2_000_003_072, "source_mix": {"en": 1.0}},
                  {"tokens": 3_999_996_928, "source_mix": {"en": 0.5, "code": 0.5}}]
        source = source_manifest(phase_source_mixes=phases, source_mix={})
        view = build_horizon_view(source, total_train_tokens=4_999_610_368, microbatch=4)
        kept = view.config.phase_source_mixes
        self.assertEqual([phase["tokens"] for phase in kept], [2_000_003_072, 2_999_607_296])
        self.assertEqual(sum(phase["tokens"] for phase in kept), 4_999_610_368)
        with self.assertRaisesRegex(ValueError, "do not cover"):
            _truncate_phases(phases, 7_000_000_000)

    def test_reserve_val_holdout_splits_physical_files(self):
        import numpy as np

        from production.kimi_h100_5b.horizons import reserve_val_holdout

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = np.arange(16 * 4096, dtype=np.uint16)
            tokens.tofile(root / "val.bin")
            config = TokenDataConfig(seq_len=4096, batch_size=4,
                                     total_train_tokens=4_999_610_368,
                                     source_mix={"en": 1.0})
            from terminus.integrity import file_sha256

            source = TokenManifest(config, [
                TokenShardSpec("en.bin", "en", "train", 6_000_000_000, language="en"),
                TokenShardSpec("val.bin", "en", "val", 16 * 4096, language="en",
                               sha256=file_sha256(root / "val.bin"))])
            result = reserve_val_holdout(source, data_root=root,
                                         output_dir=root / "reserved", fraction=0.75)
            prefix, holdout = result["prefix"][0], result["holdout"][0]
            self.assertEqual(prefix.num_tokens, 12 * 4096)
            self.assertEqual(holdout.num_tokens, 4 * 4096)
            loaded_prefix = np.memmap(root / "reserved" / Path(prefix.path).name,
                                      dtype=np.uint16, mode="r")
            loaded_holdout = np.memmap(root / "reserved" / Path(holdout.path).name,
                                       dtype=np.uint16, mode="r")
            self.assertTrue(np.array_equal(loaded_prefix, tokens[:12 * 4096]))
            self.assertTrue(np.array_equal(loaded_holdout, tokens[12 * 4096:]))
            reservation = result["reservation"][0]
            self.assertEqual(reservation["parent_sha256"], file_sha256(root / "val.bin"))
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                reserve_val_holdout(source, data_root=root, output_dir=root / "x", fraction=1.0)
            tiny = TokenManifest(config, [
                TokenShardSpec("en.bin", "en", "train", 6_000_000_000, language="en"),
                TokenShardSpec("tinyval.bin", "en", "val", 3 * 4096, language="en")])
            np.zeros(3 * 4096, dtype=np.uint16).tofile(root / "tinyval.bin")
            with self.assertRaisesRegex(ValueError, "no full sequence"):
                reserve_val_holdout(tiny, data_root=root, output_dir=root / "y", fraction=0.75)

    def test_view_with_reserved_prefix_specs(self):
        source = source_manifest()
        prefix_spec = TokenShardSpec("reserved/val.prefix.bin", "en", "val", 45_000_000,
                                     language="en", sha256="ef" * 32)
        view = build_horizon_view(source, total_train_tokens=4_999_610_368, microbatch=4,
                                  val_shards=[prefix_spec], val_reservation=[{"parent": "val.bin"}])
        val_specs = [spec for spec in view.shards if spec.split == "val"]
        self.assertEqual(val_specs, [prefix_spec])
        record = view.provenance["horizon_view"]
        self.assertFalse(record["shards_unchanged"])
        self.assertEqual(record["val_reservation"], [{"parent": "val.bin"}])
        train_specs = [spec for spec in view.shards if spec.split == "train"]
        self.assertEqual(len(train_specs), 2)

    def test_token_accounting_audit_proves_complete_updates(self):
        view = build_horizon_view(source_manifest(), total_train_tokens=4_999_610_368, microbatch=4)
        receipt = audit_token_accounting(view)
        self.assertEqual(receipt["status"], "aligned_complete_updates")
        self.assertEqual(receipt["total_updates"], 2384)
        self.assertEqual(receipt["valid_targets_per_update"], 2_097_152)
        self.assertEqual(receipt["padded_tail_targets"], 0)
        self.assertEqual(receipt["accumulation_steps"], 128)

    def test_token_accounting_rejects_tails_and_bad_batches(self):
        misaligned = TokenManifest(TokenDataConfig(seq_len=4096, batch_size=4,
                                                   total_train_tokens=2_101_248), [])
        with self.assertRaisesRegex(ValueError, "global batch"):
            audit_token_accounting(misaligned)
        tail = TokenManifest(TokenDataConfig(seq_len=4096, batch_size=4,
                                             total_train_tokens=4097), [])
        with self.assertRaisesRegex(ValueError, "sequence-aligned"):
            audit_token_accounting(tail)
        odd = TokenManifest(TokenDataConfig(seq_len=4096, batch_size=3,
                                            total_train_tokens=4_999_610_368), [])
        with self.assertRaisesRegex(ValueError, "microstep"):
            audit_token_accounting(odd)

    def test_cli_writes_fresh_external_view(self):
        with tempfile.TemporaryDirectory() as directory:
            from dataclasses import asdict

            root = Path(directory)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(asdict(source_manifest())))
            output = root / "view.json"
            import sys
            from unittest import mock

            argv = ["horizons", "view", "--source-manifest", str(source_path),
                    "--total-tokens", "4999610368", "--microbatch", "4", "--output", str(output)]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(horizons_main(), 0)
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(FileExistsError):
                    horizons_main()
            with mock.patch.object(sys, "argv",
                                   ["horizons", "audit", "--manifest", str(output)]):
                self.assertEqual(horizons_main(), 0)


if __name__ == "__main__":
    unittest.main()
