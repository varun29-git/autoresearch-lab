"""Small instrumentation boundary for the native full-update workload.

The workload supplies one COMPLETE optimizer update, including data movement,
all accumulation microsteps, clipping, optimizer, and cache invalidation. This
module never launches a VM and deliberately does not implement a second trainer.
See docs/speed/MEASUREMENT.md for integration and independent correctness gates.
"""
import math
import time


def measure_block(update, *, updates, synchronize, read_memory, clock=time.perf_counter):
    """Measure a pre-warmed block without injecting per-update CUDA barriers.

    update() returns the host-side count of valid training targets, derived
    from the locked input manifest. Check nonfinites inside the real trainer;
    a skipped/rejected optimizer update must raise, never count toward speed.
    Caller resets memory high-water statistics BEFORE this function.
    read_memory() returns total device peak bytes, including non-torch memory.
    """
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    synchronize()
    started = clock()
    tokens = 0
    for _ in range(updates):
        count = update()
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("an update must return its positive integer valid target count")
        tokens += count
    synchronize()
    seconds = clock() - started
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid elapsed time")
    memory = read_memory()
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise ValueError("invalid measured device memory")
    return {"seconds": seconds, "updates": updates, "valid_tokens": tokens,
            "peak_device_bytes": memory, "completed": True}
