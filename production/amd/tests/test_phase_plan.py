import math

import pytest

from production.amd.budget import forecast
from production.amd.phase_plan import plan


def test_split_is_by_tokens_and_charges_slow_qat_more_time():
    result = plan(dense_rate=55_000, qat_rate=45_000)
    dense, qat = result["dense"], result["qat"]
    assert 2 * dense["tokens"] == 3 * qat["tokens"]
    assert dense["tokens"] + qat["tokens"] == result["total_tokens"]
    assert result["qat_fraction_of_production_time"] == pytest.approx(22 / 49)
    assert result["effective_productive_tokens_per_second_per_gpu"] == pytest.approx(2_475_000 / 49)
    assert result["total_tokens"] == pytest.approx(864_086_204_081.6327, rel=2e-5)
    assert result["transition_at_global_update"] * result["global_batch_tokens"] == dense["tokens"]
    for phase in (dense, qat):
        assert phase["tokens"] % result["global_batch_tokens"] == 0
    # Independently cross-check the existing fixed-token budget calculator.
    check = forecast(dense_rate=55_000, qat_rate=45_000, target_tokens=result["total_tokens"],
                     qat_tokens=qat["tokens"], reserve_gpu_hours=480, duty_cycle=.9)
    assert check["fits_assumptions"]
    assert check["gpu_hours_required"] == pytest.approx(result["total_allocated_gpu_hours_used"])
    assert result["total_node_days"] <= 30 + 1e-10
    assert result["amd_qat_ready"] is False


@pytest.mark.parametrize("percent", [10, 20, 30, 40, 80])
def test_equal_phase_rates_and_aggregate_gpu_budget(percent):
    result = plan(dense_rate=50_000, qat_rate=50_000, qat_percent=percent)
    assert result["qat_fraction_of_production_time"] == pytest.approx(percent / 100)
    assert result["total_tokens"] == pytest.approx(855_360_000_000, rel=3e-5)
    bigger = plan(dense_rate=50_000, qat_rate=50_000, qat_percent=percent, gpus=32)
    assert bigger["total_tokens"] == result["total_tokens"]
    assert bigger["total_node_days"] == result["total_node_days"] / 4


def test_reserve_and_duty_cycle_are_not_available_for_training():
    full = plan(dense_rate=50_000, qat_rate=40_000, reserve_gpu_hours=0, duty_cycle=1)
    limited = plan(dense_rate=50_000, qat_rate=40_000)
    assert limited["total_tokens"] / full["total_tokens"] == pytest.approx(5280 * .9 / 5760, rel=3e-5)
    assert limited["dense"]["allocated_gpu_hours"] + limited["qat"]["allocated_gpu_hours"] <= 5280 + 1e-10


@pytest.mark.parametrize("override", [
    {"dense_rate": math.nan}, {"qat_rate": 0}, {"qat_rate": math.inf}, {"qat_rate": True},
    {"qat_percent": .4}, {"qat_percent": 100}, {"qat_percent": 0},
    {"gpus": 0}, {"gpus": True}, {"duty_cycle": 1.1}, {"duty_cycle": 0},
    {"reserve_gpu_hours": 5760}, {"reserve_gpu_hours": -1}, {"reserve_gpu_hours": True},
    {"global_batch_tokens": 4097}, {"global_batch_tokens": True}, {"gpu_hours": 1e-6, "reserve_gpu_hours": 0},
    {"dense_rate": 1e308, "qat_rate": 1e308},
])
def test_rejects_invalid_or_incomplete_accounting(override):
    with pytest.raises(ValueError):
        plan(**({"dense_rate": 55_000, "qat_rate": 45_000} | override))
