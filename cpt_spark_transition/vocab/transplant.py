"""Build the packed embedding. Kept rows are copied. New rows are a scaled mean.

This does not open a checkpoint. The caller passes the original embedding tensor.
"""

from __future__ import annotations

import torch


def constituent_row(weight, constituent_ids):
    """FVT mean of `constituent_ids`, rescaled to their mean norm."""
    if not constituent_ids:
        raise ValueError("a new piece has no Spark constituents")
    index = torch.tensor(list(constituent_ids), dtype=torch.long)
    rows = weight.index_select(0, index).float()
    mean = rows.mean(dim=0)
    center = mean.norm()
    if float(center) == 0.0:
        return mean.to(dtype=weight.dtype)
    target = rows.norm(dim=1).mean()
    return (mean * (target / center)).to(dtype=weight.dtype)


def assemble(weight, kept_old_ids, constituent_lists):
    """Return [len(kept) + len(new), hidden].

    `kept_old_ids` is original-row order, ids 0-13 first. Each entry of
    `constituent_lists` is the original Spark ids of one appended piece.
    """
    if weight.dim() != 2:
        raise ValueError("embedding weight must be rank 2")
    kept_index = torch.tensor(list(kept_old_ids), dtype=torch.long)
    kept = weight.index_select(0, kept_index)
    if not constituent_lists:
        return kept
    new_rows = torch.stack([
        constituent_row(weight, constituents) for constituents in constituent_lists
    ])
    return torch.cat((kept, new_rows), dim=0)


def kept_rows_match(assembled, weight, kept_old_ids):
    """Bit equality of the packed prefix against the original rows."""
    kept_index = torch.tensor(list(kept_old_ids), dtype=torch.long)
    original = weight.index_select(0, kept_index)
    packed = assembled[:len(kept_old_ids)]
    if original.dtype != packed.dtype:
        return False
    if original.dtype == torch.bfloat16:
        return torch.equal(original.view(torch.int16), packed.view(torch.int16))
    return torch.equal(original, packed)
