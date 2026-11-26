"""Prove data-stream continuity from verified shard and routing metadata."""
from math import lcm

from .data import TrainingTokenRouter


def capture_training_stream(router):
    """Exclude file locations and microbatching; include every token-order choice."""
    config = router.config
    if router.phase_specs:
        spans = router.mixed_spans
        segments = [{"start": phase["start_example"], "end": phase["end_example"],
                     "schedule": list(phase["schedule"]),
                     "offsets": dict(phase["source_prefix_examples"])}
                    for phase in router.phase_specs]
    elif router.mix_schedule:
        spans = router.mixed_spans
        segments = [{"start": 0, "end": config.total_examples,
                     "schedule": list(router.mix_schedule),
                     "offsets": dict(config.source_start_examples)}]
    else:
        spans = {"main": router.main}
        segments = []
        start = 0
        if router.prefix is not None:
            spans["prefix"] = router.prefix
            start = config.prefix_examples
            segments.append({"start": 0, "end": start, "schedule": ["prefix"], "offsets": {}})
        if start < config.total_examples:
            segments.append({"start": start, "end": config.total_examples,
                             "schedule": ["main"], "offsets": {}})
    return {"schema": 1, "seq_len": config.seq_len, "total_examples": config.total_examples,
            "segments": segments,
            "sources": {name: [{"sha256": shard.spec.sha256.lower(),
                                "num_tokens": shard.num_tokens, "dtype": shard.spec.dtype}
                               for shard in span.shards]
                        for name, span in spans.items() if span is not None}}


def verify_training_continuation(parent, child, tokens_seen, *, policy):
    """Check the consumed prefix, and the remaining common plan for `continue`.

    This proof permits relocating and appending immutable shards. Repacking a
    previously used shard needs a separate content proof and is rejected here.
    No token payloads are read; both contracts must come from full validation.
    """
    if (not parent or not child or parent.get("schema") != 1 or child.get("schema") != 1):
        raise ValueError("data continuation needs verified training-stream metadata in both contracts")
    length = parent["seq_len"]
    if (child["seq_len"] != length or tokens_seen < 0 or tokens_seen % length
            or tokens_seen > parent["total_examples"] * length
            or tokens_seen >= child["total_examples"] * length):
        raise ValueError("data continuation has an incompatible sequence length or token cursor")
    if policy not in {"continue", "phase_mix"}:
        raise ValueError("continuation proof requires continue or phase_mix")
    compared = (min(parent["total_examples"], child["total_examples"])
                if policy == "continue" else tokens_seen // length)
    pi = ci = position = 0
    required = {}
    while position < compared:
        ps, cs = parent["segments"][pi], child["segments"][ci]
        if not ps["start"] <= position < ps["end"] or not cs["start"] <= position < cs["end"]:
            raise ValueError("data continuation has a gap in its routing segments")
        end = min(ps["end"], cs["end"], compared)
        p_schedule, c_schedule = ps["schedule"], cs["schedule"]
        p_offset, c_offset = position - ps["start"], position - cs["start"]
        # Routing repeats every 1 or 1,000 examples in the current loader.
        period = lcm(len(p_schedule), len(c_schedule))
        for step in range(min(end - position, period)):
            if p_schedule[(p_offset + step) % len(p_schedule)] != c_schedule[(c_offset + step) % len(c_schedule)]:
                raise ValueError("data continuation changed source order or mixture before the allowed boundary")
        for source in set(p_schedule):
            p_before = TrainingTokenRouter._source_count_before(source, p_offset, p_schedule)
            p_after = TrainingTokenRouter._source_count_before(source, end - ps["start"], p_schedule)
            if p_after == p_before:
                continue
            p_index = ps["offsets"].get(source, 0) + p_before
            c_index = cs["offsets"].get(source, 0) + TrainingTokenRouter._source_count_before(source, c_offset, c_schedule)
            if p_index != c_index:
                raise ValueError("data continuation changed a source offset, repeating or skipping examples")
            # One extra token supplies the final next-token target.
            required[source] = max(required.get(source, 0), (p_index + p_after - p_before) * length + 1)
        position = end
        pi += int(position == ps["end"])
        ci += int(position == cs["end"])
    for source, needed in required.items():
        p_shards, c_shards = parent["sources"][source], child["sources"].get(source, [])
        covered = 0
        for index, p_shard in enumerate(p_shards):
            if covered >= needed:
                break
            if index >= len(c_shards) or p_shard != c_shards[index]:
                raise ValueError("data continuation changed or reordered a required training shard")
            covered += p_shard["num_tokens"]
        if covered < needed:
            raise ValueError("data continuation lacks the required shard prefix")
    return {"policy": policy, "verified_prefix_tokens": compared * length,
            "consumed_tokens": tokens_seen, "required_sources": sorted(required)}


def verify_packed_continuation(parent, child, state, *, policy):
    """Packed token counts do not define sequence cursors; preserve both clocks.

    Appending phases leaves each prior finite permutation unchanged. Editing a
    consumed phase (even just appending a shard to its permutation) is rejected.
    A separate next phase can rebase its local cursor while retaining the actual
    lifetime label counter, including a partially filled final dense update.
    """
    if (not parent or not child or parent.get("schema") != "packed_documents_v1" or
            child.get("schema") != "packed_documents_v1" or parent["seq_len"] != child["seq_len"]):
        raise ValueError("packed continuation requires verified matching packed streams")
    slots, seen = state.get("input_slots_seen", -1), state["tokens_seen"]
    length = parent["seq_len"]
    if not 0 <= slots <= parent["total_examples"]*length or slots % length:
        raise ValueError("invalid packed input-slot cursor")
    if policy == "packed_phase":
        if (slots != parent["total_examples"]*length or
                seen != parent.get("tokens_before", 0)+sum(p["scored_targets"] for p in parent["phases"]) or
                child.get("tokens_before", 0) != seen):
            raise ValueError("next packed phase needs a completed parent and matching lifetime label offset")
        old_payloads = {s["payload"] for p in parent["phases"] for s in p["shards"]}
        new_payloads = {s["payload"] for p in child["phases"] for s in p["shards"]}
        if old_payloads & new_payloads:
            raise ValueError("next packed phase repeats a parent payload")
        return dict(policy=policy, consumed_tokens=seen, local_input_cursor_reset=True,
                    completed_parent_input_slots=slots)
    if policy not in {"continue", "phase_mix"} or parent.get("tokens_before", 0) != child.get("tokens_before", 0):
        raise ValueError("incompatible packed continuation policy or lifetime offset")
    compared = parent["total_examples"] if policy == "continue" else slots//length
    end = 0
    for index, phase in enumerate(parent["phases"]):
        if end >= compared:
            break
        if index >= len(child["phases"]) or phase != child["phases"][index]:
            raise ValueError("packed continuation changed a consumed finite phase/permutation")
        end += phase["sequences"]
    if policy == "phase_mix" and slots//length != end:
        raise ValueError("packed phase_mix needs an exact phase boundary")
    if slots >= child["total_examples"]*length:
        raise ValueError("packed child does not extend the sequence cursor")
    return dict(policy=policy, verified_prefix_input_slots=end*length, consumed_tokens=seen)
