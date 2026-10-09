"""Pinned Muon update with logical row slices for fused projection weights."""

from collections import defaultdict

import torch
from muon import SingleDeviceMuonWithAuxAdam, zeropower_via_newtonschulz5


class LogicalMuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    def __init__(self, param_groups, logical_rows, batched=True):
        if len(param_groups) != len(logical_rows):
            raise ValueError("logical Muon metadata must match optimizer groups")
        # The pinned upstream constructor checks exact group keys. Add the
        # slice metadata after it has established the ordinary Muon groups.
        super().__init__(param_groups)
        for group, rows in zip(self.param_groups, logical_rows):
            if rows is None:
                continue
            if not group["use_muon"] or len(group["params"]) != 1:
                raise ValueError("logical row slices require one Muon matrix")
            parameter = group["params"][0]
            if parameter.ndim != 2 or sum(rows) != parameter.shape[0] or any(row <= 0 for row in rows):
                raise ValueError("logical row slices must partition a 2D matrix")
            group["logical_rows"] = tuple(rows)
        if not isinstance(batched, bool):
            raise ValueError("batched must be a bool")
        self.batched = batched

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.batched:
            self._step_batched()
        else:
            self._step_reference()
        return loss

    def _collect_muon_slices(self):
        """Per logical slice: exact reference momentum/Nesterov semantics, grouped for launch efficiency."""
        slices = []
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for parameter in group["params"]:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                state = self.state[parameter]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                momentum = state["momentum_buffer"]
                start = 0
                for rows in group.get("logical_rows", (parameter.shape[0],)):
                    stop = start + rows
                    slices.append({"parameter": parameter, "momentum": momentum,
                                   "start": start, "stop": stop, "lr": group["lr"],
                                   "weight_decay": group["weight_decay"],
                                   "beta": group["momentum"]})
                    start = stop
        return slices

    def _step_batched(self):
        slices = self._collect_muon_slices()
        # momentum.lerp_(grad, 1 - beta); grad.lerp_(momentum, beta) (Nesterov),
        # identical elementwise math to the reference loop, grouped by beta.
        by_beta = defaultdict(list)
        for item in slices:
            by_beta[item["beta"]].append(item)
        for beta, members in by_beta.items():
            momenta = [m["momentum"][m["start"]:m["stop"]] for m in members]
            grads = [m["parameter"].grad[m["start"]:m["stop"]] for m in members]
            torch._foreach_lerp_(momenta, grads, 1 - beta)
            torch._foreach_lerp_(grads, momenta, beta)
        # Newton-Schulz: stack same-shape slices; the pinned iteration already
        # normalizes per matrix (dim=(-2, -1)) and applies the orientation rule
        # per trailing shape, so one batched call equals per-slice calls up to
        # floating-point reduction order (class E).
        by_shape = defaultdict(list)
        for item in slices:
            grad = item["parameter"].grad
            by_shape[(item["stop"] - item["start"], grad.shape[1])].append(item)
        for (rows, cols), members in sorted(by_shape.items()):
            stacked = torch.stack([m["parameter"].grad[m["start"]:m["stop"]] for m in members])
            update = zeropower_via_newtonschulz5(stacked, steps=5)
            update *= max(1, rows / cols) ** 0.5
            by_rate = defaultdict(list)
            for member in members:
                by_rate[(member["lr"], member["weight_decay"])].append(member)
            updates = list(update.float().unbind(0))
            positions = {id(m): index for index, m in enumerate(members)}
            for (lr, weight_decay), group_members in by_rate.items():
                weights = [m["parameter"][m["start"]:m["stop"]] for m in group_members]
                group_updates = [updates[positions[id(m)]] for m in group_members]
                torch._foreach_mul_(weights, 1 - lr * weight_decay)
                torch._foreach_add_(weights, group_updates, alpha=-lr)
        self._step_adam()

    def _step_adam(self):
        """Auxiliary Adam groups: reference adam_update math via foreach launches."""
        for group in self.param_groups:
            if group["use_muon"]:
                continue
            params = []
            for parameter in group["params"]:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                state = self.state[parameter]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] = 0
                state["step"] += 1
                params.append((parameter, state))
            beta1, beta2 = group["betas"]
            # Every parameter in a group shares the step count; verify rather than assume.
            steps = {state["step"] for _, state in params}
            if len(steps) != 1:
                raise RuntimeError("aux Adam parameters diverged in step count")
            step = steps.pop()
            weights = [parameter for parameter, _ in params]
            grads = [parameter.grad for parameter, _ in params]
            buf1s = [state["exp_avg"] for _, state in params]
            buf2s = [state["exp_avg_sq"] for _, state in params]
            torch._foreach_lerp_(buf1s, grads, 1 - beta1)
            torch._foreach_lerp_(buf2s, torch._foreach_mul(grads, grads), 1 - beta2)
            buf1c = torch._foreach_div(buf1s, 1 - beta1 ** step)
            denom = torch._foreach_sqrt(torch._foreach_div(buf2s, 1 - beta2 ** step))
            torch._foreach_add_(denom, group["eps"])
            update = torch._foreach_div(buf1c, denom)
            torch._foreach_mul_(weights, 1 - group["lr"] * group["weight_decay"])
            torch._foreach_add_(weights, update, alpha=-group["lr"])

    def _step_reference(self):
        from muon import adam_update, muon_update

        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    momentum = state["momentum_buffer"]
                    start = 0
                    for rows in group.get("logical_rows", (parameter.shape[0],)):
                        stop = start + rows
                        update = muon_update(parameter.grad[start:stop], momentum[start:stop], beta=group["momentum"])
                        logical_weight = parameter[start:stop]
                        logical_weight.mul_(1 - group["lr"] * group["weight_decay"])
                        logical_weight.add_(update.reshape(logical_weight.shape), alpha=-group["lr"])
                        start = stop
            else:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(parameter.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
