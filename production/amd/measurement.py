"""Conservative slowest-rank accounting for complete real-data updates."""
import math
import statistics


def summarize(ranks, *, world_size):
    if type(world_size) is not int or world_size <= 0 or len(ranks) != world_size:
        raise ValueError("missing rank measurements")
    if {r["rank"] for r in ranks} != set(range(world_size)):
        raise ValueError("duplicate or missing rank identity")
    lengths = {len(r["records"]) for r in ranks}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("rank measurement coverage differs")
    seconds, tokens = [], []
    for index in range(next(iter(lengths))):
        rows = [r["records"][index] for r in ranks]
        counts = {r["global_tokens"] for r in rows}
        if len(counts) != 1 or type(next(iter(counts))) is not int or next(iter(counts)) <= 0:
            raise ValueError("invalid or mismatched global token counts")
        if any(not math.isfinite(r["seconds"]) or r["seconds"] <= 0 or not math.isfinite(r["loss"]) for r in rows):
            raise ValueError("nonfinite loss or invalid elapsed time")
        seconds.append(max(r["seconds"] for r in rows))
        tokens.append(next(iter(counts)))
    ordered = sorted(seconds)
    total = sum(tokens)
    return {"schema": 1, "kind": "ci1_amd_benchmark", "world_size": world_size, "updates": len(seconds),
            "global_tokens": total, "slowest_rank_seconds_sum": sum(seconds),
            "tokens_per_second_total": total / sum(seconds),
            "tokens_per_second_per_gpu": total / sum(seconds) / world_size,
            "p50_update_seconds": statistics.median(seconds),
            "p95_update_seconds": ordered[math.ceil(.95 * len(ordered)) - 1],
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in ranks),
            "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in ranks), "ranks": ranks}
