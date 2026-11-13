import pytest

from terminus.data import TokenDataConfig, TokenManifest
from terminus.packed_data import stream_spec


def artifact(name):
    return {"uri": f"s3://example/{name}", "sha256": "ab" * 32, "bytes": 16}


def shard(plan, name, sequences=2):
    index = artifact(f"{name}-index")
    return {
        "sequence_length": 4,
        "plan_sha256": plan,
        "sequences": sequences,
        "content_tokens": sequences,
        "valid_targets": sequences * 4,
        "index_offsets_bind_to_sha256": index["sha256"],
        "payload": artifact(f"{name}-payload"),
        "index": index,
        "index_offsets": artifact(f"{name}-offsets"),
        "leaf": "en/test",
    }


def phase(plan, shards, source_plans=None):
    body = {
        "schema": "ci1_packed_mixture_candidate_v1",
        "plan_sha256": plan,
        "repeats_sequences": False,
        "total_sequences": sum(item["sequences"] for item in shards),
        "content_tokens": sum(item["content_tokens"] for item in shards),
        "shards": shards,
    }
    if source_plans is not None:
        body["source_plan_sha256"] = source_plans
    return body


def manifest(train, val):
    paths = {}
    for item in train + val:
        for shard_item in item["shards"]:
            for name in ("payload", "index", "index_offsets"):
                paths[shard_item[name]["uri"]] = shard_item[name]["uri"]
    slots = train[0]["total_sequences"] * 4
    return TokenManifest(
        TokenDataConfig(seq_len=4, total_train_tokens=slots),
        [],
        provenance={
            "packing": "document_segments_v1",
            "packed_stream": {"schema": 1, "train": train, "val": val, "paths": paths},
        },
    )


def test_one_plan_still_requires_the_phase_digest():
    plan = "a" * 64
    heldout = phase("b" * 64, [shard("b" * 64, "val")])
    accepted = stream_spec(manifest([phase(plan, [shard(plan, "train")])], [heldout]))
    assert accepted["train"][0]["plan_sha256"] == plan
    with pytest.raises(ValueError, match="phase or target contract"):
        stream_spec(manifest([phase(plan, [shard("c" * 64, "other")])], [heldout]))


def test_merged_phase_accepts_only_its_declared_source_plans():
    first, second = "a" * 64, "b" * 64
    merged = phase("c" * 64, [shard(first, "left"), shard(second, "right", sequences=3)], [first, second])
    heldout = phase("d" * 64, [shard("d" * 64, "val")])
    accepted = stream_spec(manifest([merged], [heldout]))
    assert {item["plan_sha256"] for item in accepted["train"][0]["shards"]} == {first, second}
    foreign = phase("c" * 64, [shard("e" * 64, "foreign")], [first, second])
    with pytest.raises(ValueError, match="phase or target contract"):
        stream_spec(manifest([foreign], [heldout]))
    with pytest.raises(ValueError, match="source_plan_sha256"):
        stream_spec(manifest([phase("c" * 64, [shard(first, "dup")], [first, first])], [heldout]))
