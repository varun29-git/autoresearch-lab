"""CPU-only regression checks for the handoff's budget/transfer hazards."""

import json
from pathlib import Path
import tempfile
import unittest

from .plan import REPOSITORY, accumulation, make_plan, save_plan, schedule


class CampaignPlanTests(unittest.TestCase):
    def test_historical_anchor_is_distinct_and_width_transfer_applied_once(self):
        trials = {trial["id"]: trial for trial in make_plan()["mandatory_proxy_trials_in_order"]}
        self.assertEqual(list(trials), ["H", "R0", "R1", "R2", "R3"])
        self.assertEqual(trials["H"]["proxy_adamw_lr"], .006)
        self.assertEqual(trials["H"]["transferred_adamw_lr"], .002)
        self.assertEqual(trials["R0"]["transferred_adamw_lr"], .003)
        self.assertEqual(trials["R3"]["transferred_adamw_lr"], .00085)
        for trial in trials.values():
            self.assertEqual(trial["proxy_muon_lr"], trial["transferred_muon_lr"])

    def test_proxy_horizon_and_boundaries_have_no_partial_update(self):
        plan = make_plan()["proxy_schedule"]
        self.assertEqual(plan["total_tokens"], 1_998_585_856)
        self.assertEqual(plan["boundary_updates"], [238, 762, 953])
        self.assertEqual(plan["boundary_tokens"], [499_122_176, 1_598_029_824, 1_998_585_856])
        self.assertEqual(sum(plan[key] for key in ("warmup_tokens", "stable_tokens", "decay_tokens")),
                         plan["total_tokens"])

    def test_optional_batch_preserves_tokens_but_doubles_updates(self):
        plan = make_plan()
        full_batch = plan["proxy_schedule"]
        half_batch = plan["optional_batch_trial"]["schedule"]
        self.assertEqual(half_batch["boundary_tokens"], full_batch["boundary_tokens"])
        self.assertEqual(half_batch["boundary_updates"], [2 * n for n in full_batch["boundary_updates"]])
        self.assertFalse(plan["optional_batch_trial"]["included_in_forecast"])

    def test_one_full_size_confirmation_has_its_own_short_schedule(self):
        confirmation = make_plan()["full_confirmation"]
        self.assertEqual(confirmation["number_of_candidates"], 1)
        self.assertEqual(confirmation["selection"], "pending")
        self.assertEqual(confirmation["schedule"]["total_tokens"], 599_785_472)
        self.assertEqual(confirmation["schedule"]["boundary_updates"], [29, 229, 286])

    def test_gpu_count_does_not_multiply_global_batch(self):
        self.assertEqual(accumulation(1, 4), 128)
        self.assertEqual(accumulation(1, 1), 512)
        self.assertEqual(accumulation(8, 16), 4)
        self.assertEqual(accumulation(8, 8), 8)
        for world, micro in ((0, 4), (True, 4), (1, 3), (1, 1024)):
            with self.subTest(world=world, micro=micro), self.assertRaises(ValueError):
                accumulation(world, micro)

    def test_default_is_tight_and_slower_full_model_breaks_cap(self):
        hours = make_plan()["forecast_hours"]
        self.assertAlmostEqual(hours["total"], 22.1767376, places=5)
        self.assertTrue(hours["fits_arithmetic_cap"])
        self.assertGreater(hours["headroom"], 1.8)
        self.assertFalse(make_plan(completed_h_updates=0)["forecast_hours"]["fits_arithmetic_cap"])
        self.assertFalse(make_plan(include_optional_batch=True)["forecast_hours"]["fits_arithmetic_cap"])

    def test_inputs_cannot_hide_invalid_rates_or_expand_allocation(self):
        for kwargs in ({"proxy_tps": 0}, {"proxy_tps": float("nan")},
                       {"full_tps": float("inf")}, {"reserve_hours": -1},
                       {"cap_hours": 25}, {"proxy_tps": True},
                       {"include_optional_batch": "yes"}, {"proxy_tps": 1e-310}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plan(**kwargs)
        with self.assertRaises(ValueError):
            schedule(10, 5, 5)

    def test_no_remaining_full_size_time_has_no_fake_rate_threshold(self):
        hours = make_plan(proxy_tps=100_000)["forecast_hours"]
        self.assertFalse(hours["fits_arithmetic_cap"])
        self.assertIsNone(hours["minimum_full_tps_at_other_inputs"])

    def test_external_writer_refuses_repo_symlinks_and_overwrites(self):
        plan = make_plan()
        with self.assertRaisesRegex(ValueError, "outside"):
            save_plan(plan, REPOSITORY / "should-not-be-written.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = save_plan(plan, root / "forecast.json")
            self.assertEqual(json.loads(path.read_text()), plan)
            with self.assertRaises(FileExistsError):
                save_plan(plan, path)
            (root / "repo").symlink_to(REPOSITORY, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside"):
                save_plan(plan, root / "repo" / "should-not-be-written.json")


if __name__ == "__main__":
    unittest.main()
