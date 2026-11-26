"""CPU planning for fixed QAT, posttraining reserves and unique-data limits.

This is an arithmetic forecast, not a launch command or runtime sampler.
Rates count scored targets across the whole node and include normal cadence.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from collections.abc import Mapping

from production.artifacts import external_path
from terminus.integrity import file_sha256, write_json_atomic
from .allocation import finite_positive, fixed_qat_forecast, load, remaining


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f'{name} must be a {"positive" if positive else "nonnegative"} integer')
    return value


def _weights(weights):
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError('mix needs nonempty integer cell weights')
    for cell, weight in weights.items():
        if not isinstance(cell, str) or not cell.strip():
            raise ValueError('mix cell IDs must be nonempty strings')
        _integer(weight, 'cell weight', positive=True)
    return dict(sorted(weights.items()))


def apportion(total_tokens, weights):
    """Exact final procurement quotas; not cumulative runtime routing.

    Largest-remainder allocation is only used for a complete final budget.
    Differencing repeated allocations is not a monotonic online sampler.
    """
    _integer(total_tokens, 'total_tokens')
    weights = _weights(weights)
    denominator = sum(weights.values())
    shares = {cell: divmod(total_tokens * weight, denominator)
              for cell, weight in weights.items()}
    quotas = {cell: quotient for cell, (quotient, _) in shares.items()}
    priority = sorted(weights, key=lambda cell: (-shares[cell][1], cell))
    for cell in priority[:total_tokens - sum(quotas.values())]:
        quotas[cell] += 1
    return quotas


def unique_mix_capacity(weights, available_tokens, *, quantum=1):
    """Conservative proportional capacity. A missing cell has zero supply.

    available_tokens must already exclude consumed/reserved documents and
    ineligible origins. This function cannot certify a corpus as unique.
    No cell can borrow from another cell or repeat its documents.
    """
    weights = _weights(weights)
    _integer(quantum, 'quantum', positive=True)
    if not isinstance(available_tokens, Mapping):
        raise ValueError('available_tokens must be a cell mapping')
    for count in available_tokens.values():
        _integer(count, 'available cell tokens')
    denominator = sum(weights.values())
    limits = {cell: Fraction(available_tokens.get(cell, 0) * denominator, weight)
              for cell, weight in weights.items()}
    limit = min(limits.values())
    capacity = (limit.numerator // limit.denominator // quantum) * quantum
    quotas = apportion(capacity, weights)
    if any(quotas[cell] > available_tokens.get(cell, 0) for cell in weights):
        raise AssertionError('proportional cap exceeded a cell inventory')
    return {
        'capacity_tokens': capacity,
        'limiting_cells': [cell for cell in weights if limits[cell] == limit],
        'quotas': quotas,
        'uniqueness_verified': False,
        'assumption': 'input counts are eligible, unused and disjoint; no borrowing or epochs',
    }


def program_forecast(*, remaining_node_seconds, gpus, batch_tokens,
                     dense_cluster_tps, qat_cluster_tps, qat_tokens,
                     max_dense_tokens, baseline_pretraining_tokens,
                     remaining_reserves_node_hours, dense_tokens_seen=0,
                     qat_tokens_seen=0, dense_warmup_tokens=0,
                     dense_cooldown_tokens=0, minimum_dense_tokens=0, phase='dense'):
    """Fund all remaining phase commitments before expanding dense.

    Reserve inputs are *remaining* paid-node hours, not original envelopes.
    The allocation ledger charges elapsed wall time once, including idle time.
    Optional posttraining is also reserved until explicitly released early.
    """
    _integer(gpus, 'gpus', positive=True)
    _integer(baseline_pretraining_tokens, 'baseline_pretraining_tokens', positive=True)
    _integer(batch_tokens, 'batch_tokens', positive=True)
    _integer(minimum_dense_tokens, 'minimum_dense_tokens')
    if minimum_dense_tokens % batch_tokens:
        raise ValueError('minimum dense prefix must align with complete global updates')
    if baseline_pretraining_tokens % batch_tokens:
        raise ValueError('baseline must align with complete global updates')
    if baseline_pretraining_tokens < qat_tokens:
        raise ValueError('baseline cannot be smaller than the fixed QAT commitment')
    if not isinstance(remaining_reserves_node_hours, Mapping):
        raise ValueError('remaining reserves must be a named mapping')
    reserves = []
    for name, hours in sorted(remaining_reserves_node_hours.items()):
        if (not isinstance(name, str) or not name.strip() or isinstance(hours, bool)
                or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours < 0):
            raise ValueError('reserves need names and finite nonnegative node hours')
        reserves.append({'phase': name, 'node_hours': hours, 'gpu_hours': hours * gpus,
                         'status': 'remaining reserved ceiling; not measured workload duration'})
    reserve_seconds = sum(row['node_hours'] for row in reserves) * 3600
    forecast = fixed_qat_forecast(
        remaining_node_seconds=remaining_node_seconds, dense_tokens_seen=dense_tokens_seen,
        dense_cluster_tps=dense_cluster_tps, qat_tokens=qat_tokens,
        qat_tokens_seen=qat_tokens_seen, qat_cluster_tps=qat_cluster_tps,
        batch_tokens=batch_tokens, dense_cooldown_tokens=dense_cooldown_tokens,
        reserve_seconds=reserve_seconds, max_dense_tokens=max_dense_tokens,
        phase=phase, dense_warmup_tokens=dense_warmup_tokens)
    if forecast['dense_end_tokens'] < minimum_dense_tokens:
        raise ValueError('budget/data cap cannot fund the required dense data prefix')
    phases = [
        {'phase': 'dense', 'remaining_tokens': forecast['remaining_dense_tokens'],
         'node_hours': forecast['remaining_dense_tokens'] / dense_cluster_tps / 3600},
        {'phase': 'qat', 'remaining_tokens': forecast['qat_tokens_remaining'],
         'node_hours': forecast['qat_tokens_remaining'] / qat_cluster_tps / 3600},
    ]
    for row in phases:
        row['gpu_hours'] = row['node_hours'] * gpus
    phases.extend(reserves)
    return {
        'forecast': forecast, 'phases': phases,
        'remaining_allocation_gpu_hours': remaining_node_seconds * gpus / 3600,
        'remaining_reserved_gpu_hours': reserve_seconds * gpus / 3600,
        'planned_remaining_gpu_hours': sum(row['gpu_hours'] for row in phases),
        'unallocated_gpu_hours': forecast['slack_seconds'] * gpus / 3600,
        'extension_above_baseline_tokens': max(0, forecast['total_final_tokens']
                                                - baseline_pretraining_tokens),
        'baseline_shortfall_tokens': max(0, baseline_pretraining_tokens
                                         - forecast['total_final_tokens']),
        'status': 'planning_only; no rate acceptance, corpus certification or phase launch',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, help='External JSON inputs to program_forecast')
    parser.add_argument('--output', required=True)
    parser.add_argument('--allocation', help='Original persistent allocation ledger for a live forecast')
    parser.add_argument('--gpu-hours', type=float, default=5760., help='Scenario budget without a ledger')
    args = parser.parse_args()
    policy_path = external_path(args.policy)
    output = external_path(args.output, fresh=True)
    inputs = json.loads(policy_path.read_text())
    budget = None
    if args.allocation:
        ledger = load(args.allocation)
        if inputs['gpus'] != ledger['gpus']:
            raise ValueError('policy GPU count differs from the allocation ledger')
        budget = remaining(ledger)
        seconds = budget['remaining_node_seconds']
        identity = ledger['identity_sha256']
    else:
        finite_positive(args.gpu_hours, 'gpu_hours')
        _integer(inputs['gpus'], 'gpus', positive=True)
        seconds = args.gpu_hours / inputs['gpus'] * 3600
        identity = None
    result = program_forecast(**inputs, remaining_node_seconds=seconds)
    write_json_atomic(output, {
        'policy_sha256': file_sha256(policy_path), 'inputs': inputs,
        'allocation_identity': identity, 'elapsed_budget': budget, 'result': result,
    })
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
