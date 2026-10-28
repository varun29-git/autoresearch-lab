"""Full-state qualification telemetry: observational, complete coverage.

These probes close the gaps MONITORING.md identifies in the sampled
`terminus.stability` monitor: every optimizer group LR every update, required
-gradient presence, fused complete finite checks of parameters, gradients and
optimizer moments, logical-role update deltas with direction cosines, a
nominal-decay ledger, and read-only observation of the actual native FP8
cached operands (never a recomputation).

Isolation contract: no probe draws from the training RNG, changes weights or
moments, mutates cached FP8 operands, or touches the data cursor. Reference
branches snapshot and restore complete state around isolated updates.
Every function records its own wall time so the runner can charge measured
diagnostic overhead.
"""

from __future__ import annotations

import math
import time

import torch

# FP8 e4m3 largest finite magnitude; boundary occupancy is reported, not
# judged: dynamic block scaling can place a block maximum here legitimately.
FP8_E4M3_MAX_FINITE = 448.0


class QualificationFailure(RuntimeError):
    """A mandatory observation failed; the current attempt must stop."""


def _timed(seconds_since: float) -> float:
    return time.monotonic() - seconds_since


def group_lr_records(optimizer) -> list:
    """Every optimizer group's effective LR, role and base LR for one update."""
    records = []
    for index, group in enumerate(optimizer.param_groups):
        records.append({
            "group_index": index,
            "name": group.get("name", f"group_{index}"),
            "use_muon": bool(group.get("use_muon")),
            "lr": group["lr"],
            "base_lr": group["base_lr"],
            "weight_decay": group.get("weight_decay", 0.0),
            "logical_rows": list(group["logical_rows"]) if group.get("logical_rows") else None,
        })
    return records


def required_gradient_report(model, *, documented_absent=()) -> dict:
    """Every trainable parameter must hold a gradient unless documented absent."""
    started = time.monotonic()
    missing, checked = [], 0
    absent = set(documented_absent)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        checked += 1
        if parameter.grad is None and name not in absent:
            missing.append(name)
    return {"checked": checked, "missing": missing,
            "documented_absent": sorted(absent), "seconds": _timed(started)}


def _finite_and_count(tensor, chunk_elements: int = 1 << 24):
    """Complete finite scan; returns (all_finite, element_count)."""
    flat = tensor.detach().reshape(-1)
    count = flat.numel()
    if count == 0:
        return True, 0
    finite = True
    for offset in range(0, count, chunk_elements):
        if not bool(torch.isfinite(flat[offset:offset + chunk_elements]).all()):
            finite = False
            break
    return finite, count


def full_state_finite_report(model, optimizer) -> dict:
    """Fused/chunked complete finite coverage of parameters, gradients, moments.

    A finite sampled loss does not rule out corrupted state; this scans every
    element of every parameter, gradient and optimizer-state tensor.
    """
    started = time.monotonic()
    failures, tensors, elements = [], 0, 0
    for name, parameter in model.named_parameters():
        ok, count = _finite_and_count(parameter)
        tensors, elements = tensors + 1, elements + count
        if not ok:
            failures.append(f"parameter:{name}")
        if parameter.grad is not None:
            ok, count = _finite_and_count(parameter.grad)
            tensors, elements = tensors + 1, elements + count
            if not ok:
                failures.append(f"gradient:{name}")
    for key, state in optimizer.state_dict()["state"].items():
        for slot, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point():
                ok, count = _finite_and_count(value)
                tensors, elements = tensors + 1, elements + count
                if not ok:
                    failures.append(f"optimizer_state:{key}:{slot}")
    return {"finite": not failures, "failures": failures,
            "tensors_checked": tensors, "elements_checked": elements,
            "seconds": _timed(started)}


def require_healthy_state(model, optimizer, *, documented_absent=()) -> dict:
    """Fail closed on missing required gradients or any nonfinite state."""
    gradients = required_gradient_report(model, documented_absent=documented_absent)
    if gradients["missing"]:
        raise QualificationFailure(
            "missing required gradients: " + ", ".join(gradients["missing"][:8]))
    finite = full_state_finite_report(model, optimizer)
    if not finite["finite"]:
        raise QualificationFailure(
            "nonfinite model/optimizer state: " + ", ".join(finite["failures"][:8]))
    return {"gradients": gradients, "finite": finite}


_MUON_SLICE_ROLES = {
    ".attn.qkv_proj.weight": ("q", "k", "v"),
    ".ffn.gate_up_proj.weight": ("gate", "up"),
}


class RoleDeltaTracker:
    """Logical-role weight/gradient/update statistics across diagnostic points.

    Muon matrices are split into their logical Q/K/V and gate/up slices, and
    each AdamW group is one role. Direction cosines compare the current
    diagnostic delta against the previous one; the compared update ids are
    recorded. Ratios use an explicit denominator floor so initially near-zero
    gains cannot produce unbounded ratios; absolute deltas are always kept.
    """

    def __init__(self, optimizer, *, denominator_floor: float = 1e-12):
        if not math.isfinite(denominator_floor) or denominator_floor <= 0:
            raise ValueError("denominator floor must be positive and finite")
        self.floor = denominator_floor
        self.roles = []
        for index, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group_{index}")
            params = list(group["params"])
            if group.get("use_muon"):
                if len(params) != 1:
                    raise ValueError("Muon groups must hold exactly one logical matrix")
                rows = group.get("logical_rows") or (params[0].shape[0],)
                slice_labels = next((labels for suffix, labels in _MUON_SLICE_ROLES.items()
                                     if name.endswith(suffix)), None)
                offset = 0
                for position, rows_in_slice in enumerate(rows):
                    label = slice_labels[position] if slice_labels else (name if len(rows) == 1
                                                                         else f"slice_{position}")
                    self.roles.append({
                        "role": f"{name}:{label}" if len(rows) > 1 else name,
                        "group_index": index, "param": params[0],
                        "rows": slice(offset, offset + rows_in_slice),
                        "use_muon": True,
                    })
                    offset += rows_in_slice
            else:
                self.roles.append({"role": name, "group_index": index, "params": params,
                                   "rows": None, "use_muon": False})
        self._before = None
        self._previous_delta = None
        self._previous_update = None
        self.decay_log_retention = {}

    @torch.no_grad()
    def record_decay(self, optimizer) -> None:
        """Accumulate sum(log(1 - eta_t * wd)) per group; nominal decay only."""
        for index, group in enumerate(optimizer.param_groups):
            lr, wd = group["lr"], group.get("weight_decay", 0.0)
            if lr and wd:
                name = group.get("name", f"group_{index}")
                self.decay_log_retention[name] = (self.decay_log_retention.get(name, 0.0)
                                                  + math.log1p(-lr * wd))

    def _role_weight(self, role):
        if role["use_muon"]:
            return role["param"].detach()[role["rows"]].float()
        return torch.cat([p.detach().reshape(-1).float() for p in role["params"]])

    @torch.no_grad()
    def begin(self) -> None:
        """Snapshot per-role weights before a diagnostic update (CPU copies)."""
        self._before = [self._role_weight(role).cpu().clone() for role in self.roles]

    @staticmethod
    def _rms(vector) -> float:
        if vector.numel() == 0:
            return 0.0
        return float(vector.pow(2).mean().sqrt())

    @torch.no_grad()
    def finish(self, update_id: int) -> dict:
        """Compare after-update weights against the begin() snapshot."""
        if self._before is None:
            raise RuntimeError("begin() must snapshot weights before the update")
        started = time.monotonic()
        records, deltas = [], []
        for role, before in zip(self.roles, self._before):
            after = self._role_weight(role).cpu()
            delta = after - before
            grads = ([role["param"].grad[role["rows"]].detach().float()]
                     if role["use_muon"] and role["param"].grad is not None
                     else [p.grad.detach().float() for p in role["params"] if p.grad is not None])
            grad_rms = (self._rms(torch.cat([g.reshape(-1).cpu() for g in grads]))
                        if grads else None)
            weight_rms_after = self._rms(after)
            update_rms = self._rms(delta)
            record = {
                "role": role["role"], "use_muon": role["use_muon"],
                "weight_rms_before": self._rms(before),
                "weight_rms_after": weight_rms_after,
                "gradient_rms_after_clip": grad_rms,
                "update_rms": update_rms,
                "update_to_weight_rms": update_rms / max(weight_rms_after, self.floor),
                "denominator_floor": self.floor,
                "update_id": update_id,
            }
            records.append(record)
            deltas.append(delta.reshape(-1))
        if self._previous_delta is not None:
            for record, delta, previous in zip(records, deltas, self._previous_delta):
                numerator = float((delta * previous).sum())
                denominator = float(delta.norm() * previous.norm())
                record["direction_cosine"] = (numerator / denominator
                                              if denominator > 0 else None)
                record["compared_updates"] = [self._previous_update, update_id]
        self._previous_delta, self._previous_update = deltas, update_id
        self._before = None
        return {"schema": 1, "update_id": update_id, "roles": records,
                "decay_log_retention": dict(self.decay_log_retention),
                "seconds": _timed(started)}


def _operand_tensor_stats(prefix: str, value, out: dict) -> None:
    """Read-only recursive stats over an FP8 operand structure."""
    if torch.is_tensor(value):
        entry = {"dtype": str(value.dtype), "shape": list(value.shape)}
        if value.is_floating_point() or value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            flat = value.detach().reshape(-1)
            as_float = flat.float()
            entry["zero_fraction"] = float((as_float == 0).float().mean()) if flat.numel() else 0.0
            entry["max_abs"] = float(as_float.abs().max()) if flat.numel() else 0.0
            if value.dtype == torch.float8_e4m3fn:
                entry["boundary_occupancy_fraction"] = float(
                    (as_float.abs() == FP8_E4M3_MAX_FINITE).float().mean()) if flat.numel() else 0.0
        out[prefix] = entry
    elif isinstance(value, dict):
        for key, item in value.items():
            _operand_tensor_stats(f"{prefix}.{key}", item, out)
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _operand_tensor_stats(f"{prefix}.{position}", item, out)


@torch.no_grad()
def native_fp8_cache_report(model) -> dict:
    """Observe the actual cached native FP8 weight operands; never recompute.

    Reports per-module cache presence, the cached weight version versus the
    live parameter version (a mismatch means a stale cache) and operand code/
    scale statistics. Boundary occupancy is a measurement, not a clamping
    verdict. Modules without a populated cache are reported, not quantized.
    """
    started = time.monotonic()
    modules = {}
    for name, module in model.named_modules():
        cache = getattr(module, "_fp8_weight_cache", None)
        if cache is None:
            if hasattr(module, "_fp8_weight_cache"):
                modules[name] = {"cache": "absent"}
            continue
        identity, operands = cache
        live_version = module.weight._version
        entry = {"cache": "present", "cached_weight_version": identity[2],
                 "live_weight_version": live_version,
                 "stale": identity[2] != live_version,
                 "weight_data_ptr_matches": identity[1] == module.weight.data_ptr()}
        _operand_tensor_stats("operands", operands, entry)
        modules[name] = entry
    return {"schema": 1, "modules": modules, "seconds": _timed(started)}


def snapshot_state(model, optimizer) -> dict:
    """Complete isolated copy of weights, optimizer state and RNG for branches."""
    from terminus.checkpoint import rng_state

    return {
        "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
        "optimizer": {param_key: {slot: (value.detach().clone() if torch.is_tensor(value) else value)
                                  for slot, value in state.items()}
                      for param_key, state in optimizer.state_dict()["state"].items()},
        "rng": rng_state(),
    }


def restore_state(model, optimizer, snapshot: dict) -> None:
    """Restore a snapshot taken by snapshot_state; discards branch effects."""
    from terminus.checkpoint import restore_rng_state

    model.load_state_dict(snapshot["model"])
    state = optimizer.state_dict()["state"]
    for param_key, entry in state.items():
        saved = snapshot["optimizer"].get(param_key, {})
        for slot, value in entry.items():
            if slot in saved:
                if torch.is_tensor(value):
                    value.copy_(saved[slot])
                else:
                    entry[slot] = saved[slot]
            elif torch.is_tensor(value):
                # Lazily initialized moments did not exist at snapshot time.
                value.zero_()
            else:
                entry[slot] = 0
    restore_rng_state(snapshot["rng"])


def max_state_delta(before: dict, after: dict) -> float:
    """Largest absolute elementwise difference between two model state dicts."""
    worst = 0.0
    for key, value in after.items():
        if torch.is_tensor(value) and value.is_floating_point():
            delta = (value - before[key].to(value.device)).abs().max()
            worst = max(worst, float(delta))
    return worst
