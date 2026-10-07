"""Quantize once per tensor, cache the codes: shared activation FP8 operands.

In the FP8 stack the same hidden state is re-quantized by every consumer GEMM
separately — the normed block input feeds the packed QKV projection (and, in
gated variants, the attention gate), the FFN input feeds gate/up, and backward
re-quantizes again for dgrad/wgrad. Weights already avoid this through
BitLinear's FP8 cache; activations did not, because an activation's identity
changes every microstep. Within one microstep, though, the tensor is immutable,
so one quantization can serve every consumer.

ActivationQuantCache keys quantized operands by tensor identity
(data_ptr, version, shape, strides, dtype) exactly like the weight cache keys
weight identity; in-place mutation bumps _version and invalidates the entry.
The cache is explicitly scoped: call clear() at microstep boundaries (the
training loop already does this for weight caches). Wiring it into
DeepGEMMBitLinearFunction is a model-side change scheduled after GPU
validation; this module provides the cache and its semantics.
"""
from __future__ import annotations

import torch

from terminus.quantization import quantize_fp8_directions, validate_fp8_options


class ActivationQuantCache:
    def __init__(self):
        self._entries = {}
        self.hits = 0
        self.misses = 0

    def _identity(self, values, rowwise, transposed, eps, backend):
        return (values.data_ptr(), values._version, tuple(values.shape), tuple(values.stride()),
                values.dtype, values.device, rowwise, transposed, eps, backend,
                torch.is_inference_mode_enabled())

    def directions(self, values, *, rowwise=True, transposed=False, eps=1e-5, backend="torch"):
        """FP8 codes/scales for values in the requested GEMM orientations.

        Returns quantize_fp8_directions output; repeated calls for the same
        unmutated tensor return the cached operands instead of re-quantizing.
        """
        validate_fp8_options("full", backend)
        if values.ndim < 2 or not values.is_floating_point():
            raise ValueError("activation cache requires a floating-point tensor")
        identity = self._identity(values, rowwise, transposed, eps, backend)
        entry = self._entries.get(identity)
        if entry is None:
            self.misses += 1
            flat = values.reshape(-1, values.shape[-1])
            entry = quantize_fp8_directions(flat, rowwise=rowwise, transposed=transposed,
                                            eps=eps, backend=backend)
            self._entries[identity] = entry
        else:
            self.hits += 1
        return entry

    def clear(self):
        self._entries.clear()
