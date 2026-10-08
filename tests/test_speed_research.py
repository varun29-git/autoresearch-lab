"""CPU tests for accounting and evidence integrity, not GPU performance tests."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from speed_research.measurement import measure_block
from speed_research.physics import (Shape, budget_targets, causal_pairs, compute_floor,
                                    projection, recoverable_speedup)
from speed_research.protocol import (ROOT, analyze_screen, append_receipt, digest, make_lock,
                                     read_json, validate_campaign, validate_candidate_config, validate_lock)


def campaign_fixture():
    campaign = read_json(ROOT / "campaign.template.json")
    campaign.update(status="ready", campaign_id="test-only", research_gpu_hour_cap=20,
                    trial_gpu_hour_cap=1, research_hardware="test-H100-NVL",
                    production_hardware="test-H100-SXM", production_topology="test-4x8",
                    environment_digest="test-image", quality_protocol="test-quality-protocol",
                    qat_token_fraction=0.4, global_tokens_per_update=32768)
    for key, value in campaign.items():
        if key.endswith("_sha256") and value is None:
            campaign[key] = "a" * 64
    return campaign


def evidence_fixture(lock, *, speedup=1.10):
    pairs = []
    for index in range(6):
        base = {"run_id": f"base-{index}", "seconds": 120.0, "valid_tokens": 20 * 32768,
                "updates": 20, "peak_device_bytes": 70_000_000_000,
                "billed_gpu_seconds": 150.0, "finite": True, "completed": True}
        candidate = {**base, "run_id": f"candidate-{index}", "seconds": 120 / speedup}
        pairs.append({"order": "AB" if index % 2 == 0 else "BA",
                      "baseline": base, "candidate": candidate})
    return {"schema": 1, "lock_sha256": lock["sha256"], "phase": "dense", "stage": "full_update",
            "world_size": 1, "native_sm90": True, "hardware": "test-H100-NVL",
            "baseline_commit": "a" * 40, "candidate_commit": "b" * 40, "pairs": pairs}


class PhysicsTests(unittest.TestCase):
    def test_aggregate_gpu_hours_not_wall_hours(self):
        result = budget_targets()
        self.assertAlmostEqual(result["minimum_per_gpu_tps"], 69444.4444444)
        self.assertAlmostEqual(result["target_cluster_tps"], 2962962.96296296)
        self.assertEqual(result["production_wall_hours"], 187.5)
        self.assertEqual(budget_targets(gpus=1)["minimum_per_gpu_tps"], result["minimum_per_gpu_tps"])

    def test_overhead_and_harmonic_phase_cost(self):
        result = projection(dense_tps=100000, qat_tps=80000, qat_fraction=0.4,
                            scaling_efficiency=0.9, unavailable_fraction=0.1)
        expected = 1 / (0.6 / 100000 + 0.4 / 80000)
        self.assertAlmostEqual(result["single_gpu_phase_weighted_tps"], expected)
        self.assertTrue(math.isclose(result["projected_tokens"], 6000 * 3600 * 0.81 * expected, rel_tol=1e-12))
        self.assertLess(result["single_gpu_phase_weighted_tps"], 92000)

    def test_phase_endpoints(self):
        for fraction, rate in [(0, 100), (1, 50)]:
            result = projection(dense_tps=100, qat_tps=50, qat_fraction=fraction,
                                scaling_efficiency=1, unavailable_fraction=0)
            self.assertEqual(result["effective_per_gpu_tps"], rate)

    def test_actual_causal_masks(self):
        for length in [1, 3, 17]:
            for window in [1, 2, 8, 32]:
                explicit = sum(0 <= q - k < window for q in range(length) for k in range(length))
                self.assertEqual(causal_pairs(length, window), explicit)
        self.assertEqual(causal_pairs(4096, 1024), 3670528)
        self.assertEqual(causal_pairs(4096, None), 8390656)

    def test_target_and_archived_proxy_parameter_counts(self):
        self.assertEqual(Shape().parameters, 1500456496)
        self.assertEqual(Shape(width=768, ffn=1920, heads=6, kv_heads=2).parameters, 200329264)

    def test_dense_mixed_precision_floor_not_sparse_peak(self):
        result = compute_floor()
        expected = 8095334400 / 1979e12 + (905969664 + 778705920) / 989.5e12
        self.assertAlmostEqual(result["ideal_seconds_per_token"], expected)
        self.assertAlmostEqual(result["optimistic_tokens_per_second_per_gpu"], 172617.032387, places=5)
        self.assertLess(compute_floor(hardware="h100_nvl")["optimistic_tokens_per_second_per_gpu"],
                        result["optimistic_tokens_per_second_per_gpu"])

    def test_amdahl(self):
        self.assertAlmostEqual(recoverable_speedup(.1, 2), 1 / .95)

    def test_invalid_physics_inputs(self):
        for value in [float("nan"), float("inf"), -1, True, 0]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget_targets(gpu_hours=value)
        for value in [1, 1.1, -0.1, float("nan")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget_targets(unavailable_fraction=value)
        with self.assertRaises(ValueError):
            Shape(width=2305)
        with self.assertRaises(ValueError):
            projection(dense_tps=1, qat_tps=1, qat_fraction=.4,
                       scaling_efficiency=1.1, unavailable_fraction=0)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.campaign = campaign_fixture()
        self.lock = make_lock(self.campaign)
        self.evidence = evidence_fixture(self.lock)

    def test_template_is_not_launch_ready(self):
        with self.assertRaises(ValueError):
            make_lock(read_json(ROOT / "campaign.template.json"))

    def test_architecture_shortcuts_rejected(self):
        for key, value in [("n_layers", 24), ("branch_norm", "none"), ("value_residual", "none"),
                           ("local_attention_window", 512), ("vocab_size", 32000)]:
            candidate = copy.deepcopy(self.campaign)
            candidate["model"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_campaign(candidate)

    def test_missing_phase_or_budget_controls(self):
        for key in ["qat_token_fraction", "research_gpu_hour_cap", "dense_checkpoint_sha256",
                    "qat_checkpoint_sha256", "global_tokens_per_update"]:
            candidate = copy.deepcopy(self.campaign)
            candidate[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_lock(candidate)

    def test_candidate_execution_knobs_without_architecture_shortcuts(self):
        candidate = copy.deepcopy(self.campaign["model"])
        candidate["gradient_checkpointing"] = True
        self.assertEqual(validate_candidate_config(candidate, phase="dense")["execution_overrides"],
                         ["gradient_checkpointing"])
        candidate["weight_mode"] = "ternary"
        validate_candidate_config(candidate, phase="qat")
        with self.assertRaises(ValueError):
            validate_candidate_config(candidate, phase="dense")
        candidate["n_layers"] = 24
        with self.assertRaises(ValueError):
            validate_candidate_config(candidate, phase="qat")

    def test_trial_charge_cap(self):
        for pair in self.evidence["pairs"]:
            pair["candidate"]["billed_gpu_seconds"] = 1000
        self.assertEqual(analyze_screen(self.evidence, self.lock)["reason"], "trial charge exceeds cap")

    def test_lock_changes_detected(self):
        altered = copy.deepcopy(self.lock)
        altered["campaign"]["global_tokens_per_update"] *= 2
        with self.assertRaises(ValueError):
            validate_lock(altered)
        altered = copy.deepcopy(self.lock)
        altered["checker_files"]["protocol.py"] = "c" * 64
        altered["sha256"] = digest({k: v for k, v in altered.items() if k != "sha256"})
        with self.assertRaises(ValueError):
            validate_lock(altered)

    def test_screen_pass_never_claims_production_or_quality(self):
        result = analyze_screen(self.evidence, self.lock)
        self.assertEqual(result["decision"], "screen_pass")
        self.assertAlmostEqual(result["geometric_mean_speedup"], 1.1)
        self.assertFalse(result["production_ready"])
        self.assertIn("32-GPU acceptance", result["remaining_gates"])

    def test_near_tie_is_unresolved(self):
        self.assertEqual(analyze_screen(evidence_fixture(self.lock, speedup=1.001), self.lock)["decision"],
                         "unresolved")

    def test_nonfinite_or_failed_update_cannot_pass(self):
        for field in ["finite", "completed"]:
            evidence = copy.deepcopy(self.evidence)
            evidence["pairs"][0]["candidate"][field] = False
            self.assertEqual(analyze_screen(evidence, self.lock)["decision"], "reject")

    def test_production_memory_envelope(self):
        self.evidence["pairs"][0]["candidate"]["peak_device_bytes"] = 80_000_000_000
        self.assertEqual(analyze_screen(self.evidence, self.lock)["decision"], "reject")

    def test_clock_drift_requires_remeasurement(self):
        self.evidence["pairs"][0]["baseline"]["seconds"] *= 1.04
        self.assertEqual(analyze_screen(self.evidence, self.lock)["decision"], "remeasure")

    def test_bad_measurement_descriptors(self):
        for key, value in [("native_sm90", False), ("world_size", 32), ("hardware", "A100"),
                           ("lock_sha256", "0" * 64), ("stage", "forward_only"), ("phase", "other")]:
            evidence = copy.deepcopy(self.evidence)
            evidence[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze_screen(evidence, self.lock)

    def test_short_duplicate_or_manipulated_blocks(self):
        for key, value in [("seconds", 1), ("seconds", float("nan")), ("updates", 1),
                           ("valid_tokens", 100000000), ("run_id", "base-0"), ("billed_gpu_seconds", 1)]:
            evidence = copy.deepcopy(self.evidence)
            evidence["pairs"][0]["candidate"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze_screen(evidence, self.lock)

    def test_order_and_replication_requirements(self):
        self.evidence["pairs"].pop()
        with self.assertRaises(ValueError):
            analyze_screen(self.evidence, self.lock)
        self.evidence = evidence_fixture(self.lock)
        for pair in self.evidence["pairs"]:
            pair["order"] = "AB"
        with self.assertRaises(ValueError):
            analyze_screen(self.evidence, self.lock)

    def test_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for value in ['{"rate": NaN}', '{"rate": 1, "rate": 2}']:
                path.write_text(value)
                with self.assertRaises(ValueError):
                    read_json(path)

    def test_ledger_retains_failures_rejects_rewrite_and_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            first = append_receipt(path, {"trial_id": "failed", "outcome": "crash", "billed_gpu_hours": .01})
            second = append_receipt(path, {"trial_id": "next", "outcome": "unresolved", "billed_gpu_hours": .1})
            self.assertEqual(second["previous"], first["sha256"])
            with self.assertRaises(ValueError):
                append_receipt(path, {"trial_id": "next", "outcome": "crash", "billed_gpu_hours": .1})
            path.write_text(path.read_text().replace('"crash"', '"screen_pass"'))
            with self.assertRaises(ValueError):
                append_receipt(path, {"trial_id": "third", "outcome": "crash", "billed_gpu_hours": .1})


class TimerTests(unittest.TestCase):
    def test_entire_block_synchronized_without_per_update_barriers(self):
        events = []
        times = iter([10, 12])
        def update():
            events.append("update")
            return 32768
        result = measure_block(update, updates=3, synchronize=lambda: events.append("sync"),
                               read_memory=lambda: 100, clock=lambda: next(times))
        self.assertEqual(events, ["sync", "update", "update", "update", "sync"])
        self.assertEqual(result["valid_tokens"], 3 * 32768)
        self.assertEqual(result["seconds"], 2)
        self.assertNotIn("finite", result)  # only the real trainer/verifier may attest this

    def test_failed_update_does_not_return_a_successful_block(self):
        def fail():
            raise FloatingPointError("failed real optimizer update")
        with self.assertRaises(FloatingPointError):
            measure_block(fail, updates=2, synchronize=lambda: None, read_memory=lambda: 100)
        with self.assertRaises(ValueError):
            measure_block(lambda: 0, updates=2, synchronize=lambda: None, read_memory=lambda: 100)


if __name__ == "__main__":
    unittest.main()
