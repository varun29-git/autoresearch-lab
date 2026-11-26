"""Allocation regressions for finishing reserves and scarce unique data."""
import math
import unittest

from production.amd.program_budget import apportion, program_forecast, unique_mix_capacity


class UniqueMixTests(unittest.TestCase):
    def test_exact_large_integer_quotas_preserve_small_cells(self):
        total = 2**60 + 123
        weights = {'web': 10**12, 'native': 37, 'math': 999_999_999_963}
        quotas = apportion(total, weights)
        self.assertEqual(sum(quotas.values()), total)
        self.assertGreater(quotas['native'], 0)
        self.assertEqual(quotas, apportion(total, dict(reversed(list(weights.items())))))

    def test_language_capacity_is_normalized_not_smallest_absolute_pool(self):
        result = unique_mix_capacity({'hi': 3, 'kn': 2}, {'hi': 9, 'kn': 7})
        self.assertEqual(result['capacity_tokens'], 15)
        self.assertEqual(result['limiting_cells'], ['hi'])
        self.assertEqual(result['quotas'], {'hi': 9, 'kn': 6})

    def test_missing_cell_stops_extension_without_borrowing(self):
        result = unique_mix_capacity({'web': 2, 'math': 1}, {'web': 10**12})
        self.assertEqual(result['capacity_tokens'], 0)
        self.assertEqual(result['limiting_cells'], ['math'])
        self.assertFalse(result['uniqueness_verified'])

    def test_round_down_complete_updates_without_overdrawing(self):
        result = unique_mix_capacity({'en': 7, 'hi': 3}, {'en': 100, 'hi': 41}, quantum=16)
        self.assertEqual(result['capacity_tokens'], 128)
        self.assertEqual(sum(result['quotas'].values()), 128)
        self.assertLessEqual(result['quotas']['hi'], 41)
        for weights, supply in [({'bad': True}, {}), ({'bad': 0}, {}),
                                 ({'valid': 1}, {'valid': -1})]:
            with self.assertRaises(ValueError):
                unique_mix_capacity(weights, supply)


class ProgramBudgetTests(unittest.TestCase):
    def inputs(self, **changes):
        inputs = dict(remaining_node_seconds=100., gpus=8, batch_tokens=10,
                      dense_cluster_tps=10., qat_cluster_tps=5., qat_tokens=100,
                      max_dense_tokens=1000, baseline_pretraining_tokens=500,
                      remaining_reserves_node_hours={'sft': 10 / 3600,
                                                     'dpo': 10 / 3600})
        inputs.update(changes)
        return inputs

    def test_funds_posttraining_and_slower_qat_before_dense(self):
        result = program_forecast(**self.inputs())
        self.assertEqual(result['forecast']['dense_end_tokens'], 600)
        self.assertEqual(result['forecast']['total_final_tokens'], 700)
        self.assertEqual(result['extension_above_baseline_tokens'], 200)
        self.assertAlmostEqual(result['planned_remaining_gpu_hours'], 100 * 8 / 3600)
        self.assertAlmostEqual(result['unallocated_gpu_hours'], 0.)

    def test_data_cap_reports_time_that_cannot_be_spent_on_reuse(self):
        result = program_forecast(**self.inputs(max_dense_tokens=300))
        self.assertEqual(result['forecast']['total_final_tokens'], 400)
        self.assertEqual(result['baseline_shortfall_tokens'], 100)
        self.assertAlmostEqual(result['unallocated_gpu_hours'], 30 * 8 / 3600)

    def test_remaining_ledger_and_reserves_are_not_reset_at_qat(self):
        result = program_forecast(**self.inputs(
            remaining_node_seconds=40, dense_tokens_seen=600, qat_tokens_seen=50, phase='qat'))
        self.assertEqual(result['forecast']['remaining_dense_tokens'], 0)
        self.assertEqual(result['forecast']['qat_tokens_remaining'], 50)
        self.assertAlmostEqual(result['unallocated_gpu_hours'], 10 * 8 / 3600)

    def test_insufficient_finishing_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'budget cannot fund'):
            program_forecast(**self.inputs(remaining_node_seconds=30))
        with self.assertRaisesRegex(ValueError, 'cooldown'):
            program_forecast(**self.inputs(dense_warmup_tokens=610))
        with self.assertRaisesRegex(ValueError, 'required dense data prefix'):
            program_forecast(**self.inputs(minimum_dense_tokens=700))

    def test_releasing_unused_reserve_grows_dense_not_qat(self):
        before = program_forecast(**self.inputs())
        after = program_forecast(**self.inputs(remaining_reserves_node_hours={}))
        self.assertEqual(after['forecast']['dense_end_tokens'] - before['forecast']['dense_end_tokens'], 200)
        self.assertEqual(after['forecast']['qat_tokens'], before['forecast']['qat_tokens'])

    def test_invalid_or_unaligned_inputs_fail(self):
        for changes in ({'gpus': True}, {'baseline_pretraining_tokens': 501},
                        {'remaining_reserves_node_hours': {'sft': math.nan}},
                        {'remaining_reserves_node_hours': {'sft': -1}},
                        {'qat_tokens': 101}):
            with self.assertRaises(ValueError):
                program_forecast(**self.inputs(**changes))


if __name__ == '__main__':
    unittest.main()
