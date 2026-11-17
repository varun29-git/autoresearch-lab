"""Whole-matrix Muon ownership for one node.

Each Muon matrix stays intact. Newton–Schulz runs on the owner, and the
updated weight and momentum are broadcast. Norms, the tied embedding, and
every Adam state stay replicated. A slice of one matrix is not an owner.
"""
from __future__ import annotations

import torch


def assign_whole_matrices(named_counts, world_size):
    """Greedy bin-pack of complete matrices. Larger matrices are placed first.

    named_counts is an iterable of (name, numel). Names are unique. The result
    maps every name to exactly one rank in range(world_size).
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    items = list(named_counts)
    names = [name for name, _ in items]
    if len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name, _ in items):
        raise ValueError("Muon matrix names must be unique nonempty strings")
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for _, count in items):
        raise ValueError("Muon matrix sizes must be positive integers")
    owners = {}
    loads = [0] * world_size
    for name, count in sorted(items, key=lambda item: (-item[1], item[0])):
        rank = min(range(world_size), key=lambda candidate: (loads[candidate], candidate))
        owners[name] = rank
        loads[rank] += count
    return owners


def install_owned_muon(optimizer, owners, rank):
    """Restrict this rank to the Muon matrices it owns. Adam groups are unchanged."""
    if rank not in set(owners.values()) and owners:
        # A rank may own nothing when there are fewer matrices than ranks.
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
    owned = []
    seen = set()
    for group in optimizer.param_groups:
        if not group.get("use_muon"):
            continue
        name = group.get("name")
        if name not in owners:
            raise ValueError(f"Muon group {name!r} has no owner")
        if len(group["params"]) != 1:
            raise ValueError("sharded Muon requires one matrix per group")
        seen.add(name)
        if owners[name] == rank:
            owned.append(id(group["params"][0]))
    missing = set(owners) - seen
    if missing:
        raise ValueError("owner map names a matrix the optimizer does not step: " + ", ".join(sorted(missing)))
    optimizer.set_active_muon(owned)
    return owners


def muon_parameter_owners(optimizer, world_size):
    named = []
    for group in optimizer.param_groups:
        if group.get("use_muon"):
            if len(group["params"]) != 1 or "name" not in group:
                raise ValueError("sharded Muon requires one named matrix per group")
            named.append((group["name"], group["params"][0].numel()))
    return assign_whole_matrices(named, world_size)


def _owned_pairs(optimizer, owners):
    for group in optimizer.param_groups:
        if not group.get("use_muon"):
            continue
        parameter = group["params"][0]
        state = optimizer.state[parameter]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = parameter.new_zeros(parameter.shape)
        yield owners[group["name"]], parameter, state["momentum_buffer"]


def broadcast_owned_muon(optimizer, owners, distributed):
    """Broadcast each owner's weight and momentum to the other ranks."""
    with torch.no_grad():
        for owner, parameter, momentum in _owned_pairs(optimizer, owners):
            distributed.broadcast(parameter.data, src=owner)
            distributed.broadcast(momentum, src=owner)


def copy_owned_muon(destination, source, owners, source_rank):
    """CPU stand-in for one owner's broadcast. Adam state is left untouched."""
    donated = {}
    for group in source.param_groups:
        if group.get("use_muon") and owners[group["name"]] == source_rank:
            parameter = group["params"][0]
            donated[group["name"]] = (parameter.detach(), source.state[parameter]["momentum_buffer"].detach())
    for group in destination.param_groups:
        if not group.get("use_muon") or owners[group["name"]] != source_rank:
            continue
        parameter = group["params"][0]
        weight, momentum = donated[group["name"]]
        with torch.no_grad():
            parameter.copy_(weight)
            if "momentum_buffer" not in destination.state[parameter]:
                destination.state[parameter]["momentum_buffer"] = parameter.new_zeros(parameter.shape)
            destination.state[parameter]["momentum_buffer"].copy_(momentum)
