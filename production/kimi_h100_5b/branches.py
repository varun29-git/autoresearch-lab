"""Isolated same-state/same-data reference update branches.

At declared probe points (initialization, warmup exit, pre-decay) the runner
replays exactly one complete update from an identical snapshot over identical
data, then restores the real post-update state. The branch never advances the
main token counters or data cursor: its loader is rebuilt from the recorded
cursor, its state is discarded, and its tokens are excluded from the trial.

The replay detects nondeterminism, state corruption and optimizer drift by
comparing real versus branch weights and moments within tolerances that are
frozen in the launch packet from independent controls — never widened after
seeing a candidate result. BF16 is a precision control, not an exact FP8
rounding oracle.
"""

from __future__ import annotations

import math

import torch

from . import telemetry


class BranchFailure(RuntimeError):
    """The reference branch diverged beyond the frozen tolerance."""


def _rebuild_iterator(workload, cursor: int):
    from terminus.data import PrefetchingTokenLoader

    if workload.iterator is not None:
        workload.iterator.close()
    workload.iterator = iter(PrefetchingTokenLoader(
        workload.manifest, root=workload.data_root, rank=workload.rank,
        world_size=workload.world_size, start_step=cursor,
        max_steps=workload.total_micro_steps - cursor,
        prefetch_batches=workload.training.prefetch_batches,
        device=workload.device, return_step=True))


def reference_replay(workload) -> dict:
    """Replay the next update from the current state; restore the real result.

    Returns per-tensor max deltas between the real and branch post-update
    weights and optimizer moments. The workload is left exactly as the real
    update left it: same weights, moments, RNG, counters and data cursor.
    """
    from terminus.checkpoint import rng_state

    state_before = dict(workload.state)
    cursor = state_before["next_micro_step"]
    snapshot = telemetry.snapshot_state(workload.model, workload.optimizer)

    workload.complete_update()  # real update N+1
    real_state = dict(workload.state)
    real_rng = rng_state()
    real_weights = {key: value.detach().clone()
                    for key, value in workload.model.state_dict().items()}
    real_moments = {id_key: {slot: (value.detach().clone() if torch.is_tensor(value) else value)
                             for slot, value in entry.items()}
                    for id_key, entry in workload.optimizer.state_dict()["state"].items()}

    telemetry.restore_state(workload.model, workload.optimizer, snapshot)
    workload.state = dict(state_before)
    _rebuild_iterator(workload, cursor)
    monitor, workload.monitor = workload.monitor, None  # the branch records no probes
    try:
        workload.complete_update()  # branch update: same state, same data, same code
    finally:
        workload.monitor = monitor
    branch_weights = workload.model.state_dict()
    branch_moments = workload.optimizer.state_dict()["state"]

    weight_delta = telemetry.max_state_delta(
        {key: value for key, value in real_weights.items()},
        {key: value for key, value in branch_weights.items()})
    moment_delta = 0.0
    for id_key, entry in branch_moments.items():
        for slot, value in entry.items():
            if torch.is_tensor(value) and value.is_floating_point():
                reference = real_moments[id_key][slot]
                moment_delta = max(moment_delta,
                                   float((value - reference.to(value.device)).abs().max()))

    # Discard the branch: restore the real post-update state, RNG and cursor.
    telemetry.restore_state(workload.model, workload.optimizer,
                            {"model": real_weights, "optimizer": real_moments, "rng": real_rng})
    workload.state = real_state
    _rebuild_iterator(workload, real_state["next_micro_step"])
    return {"schema": 1, "replayed_update": real_state["updates"],
            "max_weight_delta": weight_delta, "max_moment_delta": moment_delta,
            "branch_tokens_excluded": workload.update_tokens,
            "note": "tolerances are frozen in the launch packet from independent controls"}


def require_within_tolerance(report: dict, *, max_weight_delta: float,
                             max_moment_delta: float) -> None:
    """Fail closed when the replay exceeds the frozen native tolerances."""
    for name, bound in (("max_weight_delta", max_weight_delta),
                        ("max_moment_delta", max_moment_delta)):
        value = report[name]
        if not math.isfinite(value) or value > bound:
            raise BranchFailure(f"reference replay {name} {value} exceeds frozen {bound}")
