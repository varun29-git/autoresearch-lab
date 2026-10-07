"""Idea 1: one FP32 owner per projection, FP8 replicas on every other rank.

Today every rank holds each projection's FP32 master, FP32 gradient, Muon
momentum and both FP8 weight orientations, and DDP all-reduces FP32 gradients.
Under this scheme each projection matrix has exactly one owner rank:

1. Every rank finishes its local FP32 weight gradient (microstep accumulation
   as usual).
2. Global clipping: local sums of squares are all-reduced (divided by
   world_size**2 to match DDP's averaged-gradient semantics), and the clip
   coefficient is applied to local gradients before reduction. Scaling commutes
   with the sum, so this equals clipping the globally averaged gradient.
3. Each rank reduces its local gradient to the projection's owner
   (reduce-to-one, not all-reduce).
4. The owner divides by world_size, applies Muon once (pinned Newton-Schulz,
   batched across equal-shaped slices via update_fp8_pipeline.batched_muon_update),
   then runs produce_weight_operands: one scale pass, both FP8 layouts.
5. The owner broadcasts the FP8 operand bundle once; non-owners install the
   replicas and never hold the FP32 master or momentum for that matrix.

Ring volume per parameter: all-reduce FP32 costs 2*(w-1)/w*4 bytes; this scheme
costs (w-1)/w*4 (gradient reduce) + (w-1)/w*(codes + scales) for the replica
broadcast — roughly 25% less, see ring_volume_bytes. Replicated optimizer state
drops from ~14 bytes/parameter on every rank to ~14/world + replica bytes;
replicated_state_bytes gives the exact accounting.

Non-matrix parameters (norms, embeddings, value lambdas) stay on the ordinary
replicated path; this optimizer owns only the 2D projections. Dense FP8 and
ternary QAT weight modes share the same flow. Numerical parity with
LogicalMuonWithAuxAdam at world_size=1 is covered by the test suite; multi-rank
NCCL validation runs on the GPU (optimizations/gpu_check.py).
"""
from __future__ import annotations

import torch

from terminus.optim import muon_update_scale_for_param

from .update_fp8_pipeline import batched_muon_update, produce_weight_operands


class OwnerLayout:
    """Greedy parameter-count-balanced assignment of projections to owner ranks."""

    def __init__(self, named_shapes, world_size):
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise ValueError("owner layout requires a positive integer world size")
        if not named_shapes:
            raise ValueError("owner layout requires at least one projection")
        self.world_size = world_size
        self.owner_of = {}
        loads = [0] * world_size
        ordered = sorted(named_shapes.items(), key=lambda item: -self._numel(item[1]))
        for name, shape in ordered:
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError("owner layout covers 2D projection matrices only")
            rank = loads.index(min(loads))
            self.owner_of[name] = rank
            loads[rank] += self._numel(shape)
        self.loads = loads

    @staticmethod
    def _numel(shape):
        return shape[0] * shape[1]

    def owned_by(self, rank):
        return [name for name, owner in self.owner_of.items() if owner == rank]


class SingleProcessCollectives:
    """world_size=1 collectives: every operation is already complete."""

    world_size = 1
    rank = 0

    def reduce_sum_to(self, tensor, dst):
        if dst != 0:
            raise ValueError("single-process collectives only have rank 0")
        return tensor

    def broadcast_from(self, tensor, src):
        if src != 0:
            raise ValueError("single-process collectives only have rank 0")
        return tensor

    def all_reduce_sum(self, tensor):
        return tensor


class TorchDistCollectives:
    """torch.distributed backend (NCCL or gloo) for the owner scheme."""

    def __init__(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("TorchDistCollectives requires an initialized process group")
        self._dist = dist
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def reduce_sum_to(self, tensor, dst):
        self._dist.reduce(tensor, dst=dst, op=self._dist.ReduceOp.SUM)
        return tensor

    def broadcast_from(self, tensor, src):
        self._dist.broadcast(tensor, src=src)
        return tensor

    def all_reduce_sum(self, tensor):
        self._dist.all_reduce(tensor, op=self._dist.ReduceOp.SUM)
        return tensor


def _bundle_buffers(shape, transposed, block_size, device):
    rows, width = shape
    grid = ((rows + block_size - 1) // block_size, (width + block_size - 1) // block_size)
    buffers = [torch.empty(shape, device=device, dtype=torch.uint8),
               torch.empty(grid, device=device, dtype=torch.float32)]
    if transposed:
        buffers += [torch.empty((width, rows), device=device, dtype=torch.uint8),
                    torch.empty((grid[1], grid[0]), device=device, dtype=torch.float32)]
    return buffers


def _pack_bundle(bundle):
    tensors = []
    for key in ("forward", "backward"):
        direction = bundle[key]
        if direction is None:
            continue
        codes, scales = direction
        # Codes travel as uint8: gloo has no FP8 dtype and the bits are identical.
        tensors += [codes.view(torch.uint8), scales]
    return tensors


def _unpack_bundle(buffers, transposed):
    forward = (buffers[0].view(torch.float8_e4m3fn), buffers[1])
    backward = (buffers[2].view(torch.float8_e4m3fn), buffers[3]) if transposed else None
    return {"forward": forward, "backward": backward}


class OwnerMuonOptimizer:
    """Muon over projections with a single FP32 owner rank per matrix."""

    def __init__(self, named_shapes, *, comm=None, weight_mode="dense", muon_lr=3e-3,
                 weight_decay=0.1, momentum=0.95, ns_steps=5, muon_lr_rule="sqrt_fan_in",
                 muon_fan_in_base=None, logical_rows=None, transposed_replicas=True,
                 block_size=128, device=None):
        if weight_mode not in {"ternary", "dense"}:
            raise ValueError("weight_mode must be ternary or dense")
        self.comm = comm or SingleProcessCollectives()
        self.weight_mode, self.ns_steps = weight_mode, ns_steps
        self.muon_lr, self.weight_decay, self.momentum = muon_lr, weight_decay, momentum
        self.transposed, self.block_size = transposed_replicas, block_size
        self.layout = OwnerLayout(named_shapes, self.comm.world_size)
        device = device or "cpu"
        logical_rows = logical_rows or {}
        self.slots = {}
        for name, shape in named_shapes.items():
            owner = self.layout.owner_of[name] == self.comm.rank
            scale = muon_update_scale_for_param(torch.empty(shape, device="meta"), rule=muon_lr_rule,
                                                fan_in_base=muon_fan_in_base)
            rows = logical_rows.get(name)
            if rows is not None and (any(row <= 0 for row in rows) or sum(rows) != shape[0]):
                raise ValueError(f"logical rows must partition {name}")
            self.slots[name] = {
                "shape": tuple(shape), "owner": self.layout.owner_of[name],
                "lr": muon_lr * scale, "decay": weight_decay / scale,
                "logical_rows": tuple(rows) if rows else (shape[0],),
                "master": torch.zeros(shape, device=device, dtype=torch.float32) if owner else None,
                "momentum": torch.zeros(shape, device=device, dtype=torch.float32) if owner else None,
                "grad": torch.zeros(shape, device=device, dtype=torch.float32),
                "replica": None,
            }

    @torch.no_grad()
    def initialize_masters(self, named_weights):
        """Adopt initial FP32 weights on owners, then seed every rank's replicas."""
        for name, weight in named_weights.items():
            slot = self.slots[name]
            if weight.shape != slot["shape"]:
                raise ValueError(f"initial weight shape mismatch for {name}")
            if slot["master"] is not None:
                slot["master"].copy_(weight.float())
        self._push_replicas()

    def _push_replicas(self):
        for name, slot in self.slots.items():
            owner = slot["owner"]
            if self.comm.rank == owner:
                bundle = produce_weight_operands(slot["master"], weight_mode=self.weight_mode,
                                                 transposed=self.transposed, block_size=self.block_size)
                tensors = _pack_bundle(bundle)
            else:
                tensors = _bundle_buffers(slot["shape"], self.transposed, self.block_size,
                                          slot["grad"].device)
            for tensor in tensors:
                self.comm.broadcast_from(tensor, owner)
            slot["replica"] = _unpack_bundle(tensors, self.transposed)

    def clip_coefficient(self, max_norm):
        """Global clip coefficient matching DDP averaged-gradient semantics."""
        device = next(iter(self.slots.values()))["grad"].device
        local = torch.zeros((), dtype=torch.float64, device=device)
        for slot in self.slots.values():
            local += slot["grad"].double().square().sum()
        self.comm.all_reduce_sum(local)
        global_norm = (local / self.comm.world_size ** 2).sqrt().item()
        return min(1.0, max_norm / (global_norm + 1e-6))

    @torch.no_grad()
    def step(self, named_grads=None, *, max_norm=None):
        """Reduce local gradients to owners, update masters, broadcast replicas.

        named_grads: optional {name: local FP32 gradient} to accumulate into the
        slot buffers before the step (otherwise use whatever step_grads stored).
        """
        if named_grads is not None:
            for name, grad in named_grads.items():
                slot = self.slots[name]
                if grad.shape != slot["shape"]:
                    raise ValueError(f"gradient shape mismatch for {name}")
                slot["grad"].copy_(grad.float())
        if max_norm is not None:
            coefficient = self.clip_coefficient(max_norm)
            for slot in self.slots.values():
                slot["grad"].mul_(coefficient)
        owned_slices = []
        for name, slot in self.slots.items():
            self.comm.reduce_sum_to(slot["grad"], slot["owner"])
            if slot["master"] is None:
                continue
            slot["grad"].div_(self.comm.world_size)
            start = 0
            for rows in slot["logical_rows"]:
                stop = start + rows
                owned_slices.append((slot, start, stop))
                start = stop
        updates = batched_muon_update([slot["grad"][start:stop] for slot, start, stop in owned_slices],
                                      [slot["momentum"][start:stop] for slot, start, stop in owned_slices],
                                      beta=self.momentum, ns_steps=self.ns_steps)
        for (slot, start, stop), update in zip(owned_slices, updates):
            master = slot["master"][start:stop]
            master.mul_(1 - slot["lr"] * slot["decay"])
            master.add_(update, alpha=-slot["lr"])
        self._push_replicas()
        for slot in self.slots.values():
            slot["grad"].zero_()

    def state_dict(self):
        """Optimizer state held by this rank: FP32 master and momentum for its
        owned projections only. A checkpoint gathers every rank's dict."""
        return {name: {"master": slot["master"].clone(), "momentum": slot["momentum"].clone()}
                for name, slot in self.slots.items() if slot["master"] is not None}

    @torch.no_grad()
    def load_state_dict(self, gathered):
        """Restore from the gathered per-rank dicts and re-seed all replicas."""
        for name, slot in self.slots.items():
            if slot["master"] is None:
                continue
            state = gathered[name]
            slot["master"].copy_(state["master"])
            slot["momentum"].copy_(state["momentum"])
        self._push_replicas()


def ring_volume_bytes(total_projection_params, world_size, *, scheme="owner", block_size=128):
    """Per-step ring communication volume for projection gradients/weights.

    allreduce_fp32: ring all-reduce of FP32 gradients, 2*(w-1)/w * 4 bytes/param.
    owner: reduce-to-owner of FP32 gradients plus broadcast of both FP8 operand
    layouts (1 byte/param per orientation + FP32 block scales), each (w-1)/w.
    """
    if world_size <= 0:
        raise ValueError("ring volume requires a positive world size")
    fraction = (world_size - 1) / world_size
    if scheme == "allreduce_fp32":
        return int(2 * fraction * total_projection_params * 4)
    if scheme == "owner":
        scale_bytes = 4 / (block_size * block_size)
        replica_bytes = 2 * (1 + scale_bytes)
        return int(fraction * total_projection_params * (4 + replica_bytes))
    raise ValueError("scheme must be allreduce_fp32 or owner")


def replicated_state_bytes(total_projection_params, world_size, *, scheme="owner", block_size=128):
    """Per-GPU bytes of projection master + gradient + momentum + FP8 operands.

    allreduce_fp32: FP32 master + gradient + momentum + both FP8 orientations
    on every rank. owner: master/gradient/momentum only for the owned 1/world
    share; FP8 replicas for all projections on every rank.
    """
    scale_bytes = 4 / (block_size * block_size)
    replica_bytes = 2 * (1 + scale_bytes)
    if scheme == "allreduce_fp32":
        return int(total_projection_params * (12 + replica_bytes))
    if scheme == "owner":
        return int(total_projection_params * (12 / world_size + replica_bytes))
    raise ValueError("scheme must be allreduce_fp32 or owner")
