"""Validated packed-document metadata shared by all attention layers.

Build on the CPU, transfer once with the batch. Native attention uses linear
storage (unpadded tokens and cumulative lengths), never a quadratic mask.
One compound translation pair is one segment, just like the packer describes.
"""
from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass(frozen=True)
class DocumentLayout:
    positions: torch.Tensor
    segment_ids: torch.Tensor
    token_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    spans: tuple
    max_segment_length: int

    @classmethod
    def from_cpu(cls, positions, segment_ids):
        positions = torch.as_tensor(positions, dtype=torch.long)
        segment_ids = torch.as_tensor(segment_ids, dtype=torch.long)
        if positions.device.type != "cpu" or segment_ids.device.type != "cpu":
            raise ValueError("build document layouts on CPU before device transfer")
        if positions.ndim != 2 or positions.shape != segment_ids.shape or min(positions.shape) <= 0:
            raise ValueError("positions and segments must have matching nonempty [batch, length] shapes")
        spans, indices, lengths = [], [], []
        length = positions.shape[1]
        p, s = positions.numpy(), segment_ids.numpy()
        for row in range(len(s)):
            valid = int(np.count_nonzero(s[row] >= 0))
            if np.any(s[row, :valid] < 0) or np.any(s[row, valid:] != -1) or np.any(p[row, valid:] != 0):
                raise ValueError("padding must be a -1 segment / zero-position tail")
            starts = np.r_[0, np.flatnonzero(np.diff(s[row, :valid])) + 1, valid] if valid else [0]
            for segment, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
                start, end = int(start), int(end)
                if np.any(s[row, start:end] != segment) or not np.array_equal(p[row, start:end], np.arange(end-start)):
                    raise ValueError("segments must be contiguous, numbered from zero, with reset positions")
                spans.append((row, start, end))
                indices.append(np.arange(row*length+start, row*length+end, dtype=np.int64))
                lengths.append(end-start)
        cumulative = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
        if cumulative[-1] >= 2**31:
            raise ValueError("variable-length attention exceeds int32 cumulative-length capacity")
        return cls(positions.contiguous(), segment_ids.contiguous(),
                   torch.from_numpy(np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)),
                   torch.from_numpy(cumulative.astype(np.int32)), tuple(spans), max(lengths, default=0))

    def _map(self, function):
        return replace(self, **{name: function(getattr(self, name)) for name in
                               ("positions", "segment_ids", "token_indices", "cu_seqlens")})

    def to(self, device, non_blocking=False):
        return self._map(lambda t: t.to(device, non_blocking=non_blocking))

    def pin_memory(self):
        return self._map(lambda t: t.pin_memory())

    def record_stream(self, stream):
        for name in ("positions", "segment_ids", "token_indices", "cu_seqlens"):
            getattr(self, name).record_stream(stream)

    def validate_for(self, shape, device):
        if self.positions.shape != shape or any(getattr(self, name).device != device for name in
                ("positions", "segment_ids", "token_indices", "cu_seqlens")):
            raise ValueError("document layout must match batch shape and device")


def reference_attention(query, key, value, layout, *, local_window=None):
    from .kernels import sdpa_reference_attention

    # Keep an autograd dependency even for an entirely padded DDP rank.
    output = query * 0 + (key.sum() + value.sum()) * 0
    for row, start, end in layout.spans:
        output[row:row+1, :, start:end] = sdpa_reference_attention(
            query[row:row+1, :, start:end], key[row:row+1, :, start:end],
            value[row:row+1, :, start:end], local_window=local_window)
    return output
