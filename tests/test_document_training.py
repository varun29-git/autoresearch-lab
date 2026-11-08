import contextlib
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch

from production.document_packing import ShardWriter, fragments
from production.packed_mixture import mixture_candidate
from terminus.data import TokenDataConfig, TokenManifest, TokenShardSpec, PrefetchingTokenLoader
from terminus.document_attention import DocumentLayout
from terminus.integrity import validate_manifest
from terminus.model import BitNetLlama
from terminus.optim import OptimizerConfig
from terminus.packed_data import PackedTokenBatcher
from terminus.train import TrainConfig, run_training, evaluate
from tests.helpers import tiny_config, checkpoint_payload


def packed_fixture(root, *, examples=5, batch_size=1, accumulation=2, offset=0):
    root.mkdir(parents=True, exist_ok=True)
    paths, catalog, phases = {}, [], {}
    for split, number in (("train", examples), ("val", 3)):
        writer = ShardWriter(root, split, length=8, write_offsets=True)
        for i in range(number):
            # Two isolated documents per sequence; unequal label/padding counts.
            tokens = np.asarray([40+offset+i%7, 41+offset+i%7, 42+offset+i%7][:1+i%3])
            a = list(fragments(tokens, {"id": f"{offset}-{split}-{i}"}, length=8))
            b = list(fragments(np.asarray([55+offset+i%5]), {"id": f"{offset}-{split}-b-{i}"}, length=8))
            writer.add(a+b)
        shard = writer.close()
        digest = ("a" if split == "train" else "b")*64
        shard.update(leaf="en/test", stratum="en/test", plan_sha256=digest)
        for name in ("payload", "index", "index_offsets"):
            item = shard[name]
            item["uri"] = Path(item["path"]).name
            paths[item["uri"]] = Path(item["path"]).name
        phases[split] = [mixture_candidate(dict(physical_creation_finished=True,
            all_curriculum_quotas_satisfied=True, plan_sha256=digest, shards=[shard],
            totals={"content_tokens": shard["content_tokens"]},
            results=[dict(leaf=shard["leaf"], selected=shard["content_tokens"])]))]
        catalog.append(TokenShardSpec(paths[shard["payload"]["uri"]], "en/test", split,
                       number*8, sha256=shard["payload"]["sha256"], language="en"))
    manifest = TokenManifest(TokenDataConfig(seq_len=8, batch_size=batch_size, total_train_tokens=examples*8),
        catalog, provenance={"packing": "document_segments_v1", "packed_stream":
                            dict(schema=1, paths=paths, **phases)})
    manifest.save(root/"manifest.json")
    config = tiny_config(vocab_size=64, max_seq_len=8, pad_token_id=3, weight_mode="dense", branch_norm="peri",
                         value_residual="learned", qk_norm_mode="learned_rms")
    training = TrainConfig(device="cpu", dtype="fp32", development_mode=True,
        gradient_accumulation_steps=accumulation, warmup_tokens=0, log_interval=1,
        validation_interval=10, checkpoint_interval=10, validation_batches_per_language=2)
    (root/"config.json").write_text(json.dumps(dict(model=asdict(config), training=asdict(training),
                                                  optimizer=asdict(OptimizerConfig()))))
    return config, manifest


def test_reader_prefetch_planning_resume_and_count_guard(tmp_path):
    config, manifest = packed_fixture(tmp_path/"data")
    report = validate_manifest(manifest, tmp_path/"data", config, development_mode=True)
    batcher = PackedTokenBatcher(manifest, root=tmp_path/"data", verified_files=report["packed_files"])
    with iter(PrefetchingTokenLoader(manifest, root=tmp_path/"data", return_step=True, batcher=batcher)) as iterator:
        planned = batcher.plan_update(0, 2)
        batches = [next(iterator) for _ in range(2)]
        assert sum(int(b[2].ne(-100).sum()) for b in batches) == planned[0]
        assert planned[2] == 16
        assert all(isinstance(b[3], DocumentLayout) for b in batches)
    args = (tmp_path/"data/config.json", tmp_path/"data/manifest.json", tmp_path/"data")
    with contextlib.redirect_stdout(io.StringIO()):
        first = run_training(*args, tmp_path/"resume", stop_after_updates=1)
        final = run_training(*args, tmp_path/"resume", resume=tmp_path/"resume/checkpoints")
        run_training(*args, tmp_path/"full")
    assert first["input_slots_seen"] == 16
    assert final["tokens_seen"] == manifest.scored_train_tokens < final["input_slots_seen"] == 40
    a, b = checkpoint_payload(tmp_path/"resume"), checkpoint_payload(tmp_path/"full")
    assert a["state"] == b["state"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    results = evaluate(BitNetLlama(config), manifest, tmp_path/"data", torch.device("cpu"), torch.float32, 2, 0, 1)
    assert 0 < results["en"]["tokens"] < 16


def test_two_rank_unequal_targets_and_empty_tail_matches_single_rank(tmp_path):
    packed_fixture(tmp_path/"single_data", batch_size=2)
    packed_fixture(tmp_path/"ddp_data", batch_size=1)
    with contextlib.redirect_stdout(io.StringIO()):
        run_training(tmp_path/"single_data/config.json", tmp_path/"single_data/manifest.json",
                     tmp_path/"single_data", tmp_path/"single")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1",0)); port = listener.getsockname()[1]
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
               "--master_addr=127.0.0.1", f"--master_port={port}", "--module", "terminus.train",
               "--config", str(tmp_path/"ddp_data/config.json"), "--manifest", str(tmp_path/"ddp_data/manifest.json"),
               "--data-root", str(tmp_path/"ddp_data"), "--output-dir", str(tmp_path/"ddp")]
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo"}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    single, ddp = checkpoint_payload(tmp_path/"single"), checkpoint_payload(tmp_path/"ddp")
    assert single["state"] == ddp["state"]
    for name, weight in single["model"].items():
        torch.testing.assert_close(weight, ddp["model"][name], rtol=2e-4, atol=2e-6)


def test_packed_qat_transition_retains_lifetime_clock_and_partial_dense_tail(tmp_path):
    from terminus.transition import convert_checkpoint

    _, parent = packed_fixture(tmp_path/"parent_data")
    config, child = packed_fixture(tmp_path/"child_data", offset=1)
    with contextlib.redirect_stdout(io.StringIO()):
        state = run_training(tmp_path/"parent_data/config.json", tmp_path/"parent_data/manifest.json",
                             tmp_path/"parent_data", tmp_path/"parent")
    child.provenance["packed_stream"]["tokens_before"] = state["tokens_seen"]
    child.save(tmp_path/"child_data/manifest.json")
    payload = json.loads((tmp_path/"child_data/config.json").read_text())
    payload["model"]["weight_mode"] = "ternary"
    payload["training"]["phase_start_tokens"] = state["tokens_seen"]
    (tmp_path/"child_data/config.json").write_text(json.dumps(payload))
    receipt = convert_checkpoint(tmp_path/"parent", tmp_path/"child_data/config.json",
        tmp_path/"child_data/manifest.json", tmp_path/"child_data", tmp_path/"qat", data_policy="packed_phase")
    assert receipt["child"]["state"]["tokens_seen"] == state["tokens_seen"]
    assert receipt["child"]["state"]["next_micro_step"] == 0
    assert "optimizer" not in receipt["intentional_resets"]
    with contextlib.redirect_stdout(io.StringIO()):
        final = run_training(tmp_path/"qat/qat_config.json", tmp_path/"child_data/manifest.json",
                             tmp_path/"child_data", tmp_path/"qat", resume=tmp_path/"qat/checkpoints")
    assert final["tokens_seen"] == child.scored_train_tokens
    assert final["input_slots_seen"] == 40


def test_corrupt_descriptor_total_and_stale_verified_snapshot_rejected(tmp_path):
    config, manifest = packed_fixture(tmp_path/"data")
    report = validate_manifest(manifest, tmp_path/"data", config, development_mode=True)
    shard = manifest.provenance["packed_stream"]["train"][0]["shards"][0]
    shard["valid_targets"] += 1
    with pytest.raises(ValueError, match="totals"):
        validate_manifest(manifest, tmp_path/"data", config, development_mode=True)
    shard["valid_targets"] -= 1
    payload = tmp_path/"data"/manifest.shards[0].path
    payload.touch()
    batcher = PackedTokenBatcher(manifest, root=tmp_path/"data", verified_files=report["packed_files"])
    with pytest.raises(ValueError, match="snapshot changed"):
        batcher.train_batch(0)


def test_incorrect_update_plan_cannot_change_weights_or_moments(tmp_path, monkeypatch):
    from terminus.train import prepare_training

    packed_fixture(tmp_path/"data")
    with prepare_training(tmp_path/"data/config.json", tmp_path/"data/manifest.json",
                          tmp_path/"data", tmp_path/"run") as workload:
        before = [p.detach().clone() for p in workload.model.parameters()]
        real = workload.packed_batcher.plan_update
        monkeypatch.setattr(workload.packed_batcher, "plan_update", lambda *a: (real(*a)[0]+1, *real(*a)[1:]))
        with pytest.raises(ValueError, match="optimizer update was not applied"):
            workload.complete_update()
        for a,b in zip(before, workload.model.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert not workload.optimizer.state


def test_resume_contract_is_portable_across_verified_local_file_stats(tmp_path):
    from terminus.train import make_resume_contract

    config, manifest = packed_fixture(tmp_path/"data")
    report = validate_manifest(manifest, tmp_path/"data", config, development_mode=True)
    training = TrainConfig(device="cpu", dtype="fp32", development_mode=True)
    contract = make_resume_contract(config, training, OptimizerConfig(), report, 1, torch.device("cpu"))
    moved = report | {"packed_files": {"/new-node/cache": {"inode": 999, "mtime_ns": 123}}}
    assert contract == make_resume_contract(config, training, OptimizerConfig(), moved, 1, torch.device("cpu"))


def test_finite_phase_continuation_rejects_consumed_reshuffle():
    from terminus.stream_contract import verify_packed_continuation

    phase = dict(sequences=4, scored_targets=20, seed="fixed", shards=[])
    parent = dict(schema="packed_documents_v1", seq_len=8, total_examples=4, tokens_before=0, phases=[phase])
    child = parent | {"total_examples": 8, "phases": [phase, phase | {"seed": "later"}]}
    state = dict(input_slots_seen=32, tokens_seen=20)
    assert verify_packed_continuation(parent, child, state, policy="continue")["verified_prefix_input_slots"] == 32
    changed = child | {"phases": [phase | {"seed": "changed"}, phase]}
    with pytest.raises(ValueError, match="permutation"):
        verify_packed_continuation(parent, changed, state, policy="continue")


def test_parallel_full_acceptance_has_identical_accounting(tmp_path):
    from terminus.packed_data import verify_shards

    _, manifest = packed_fixture(tmp_path/"data")
    spec = manifest.provenance["packed_stream"]
    shards = [s for split in ("train", "val") for phase in spec[split] for s in phase["shards"]]
    serial = verify_shards(shards, spec["paths"], tmp_path/"data", 64, 1)
    parallel = verify_shards(shards, spec["paths"], tmp_path/"data", 64, 2)
    assert serial == parallel


def test_staging_covers_payload_descriptor_and_offset_files(tmp_path):
    from production.amd.staging import create, verify

    config, manifest = packed_fixture(tmp_path/"data")
    for name in manifest.provenance["packed_stream"]["paths"].values():
        (tmp_path/"data"/name).chmod(0o444)
    receipt = create(manifest, tmp_path/"data", config, development_mode=True)
    assert len(receipt["files"]) == 6
    assert verify(receipt, manifest, tmp_path/"data", config, development_mode=True)["scored_train_tokens"] == manifest.scored_train_tokens
    offsets = manifest.provenance["packed_stream"]["train"][0]["shards"][0]["index_offsets"]["uri"]
    (tmp_path/"data"/offsets).chmod(0o644)
    with pytest.raises(ValueError):
        verify(receipt, manifest, tmp_path/"data", config, development_mode=True)


def test_lru_reopen_rejects_same_size_same_mtime_replacement(tmp_path):
    from production.packed_mixture import PackedMixture, PackedSequenceReader

    config, manifest = packed_fixture(tmp_path/"data")
    report = validate_manifest(manifest, tmp_path/"data", config, development_mode=True)
    spec = manifest.provenance["packed_stream"]
    shard = spec["train"][0]["shards"][0]
    reader = PackedSequenceReader(PackedMixture([shard], seed="test"),
        resolve_path=lambda uri: tmp_path/"data"/spec["paths"][uri], verified_files=report["packed_files"])
    reader.sample(0)
    reader.open.clear()
    path = tmp_path/"data"/spec["paths"][shard["payload"]["uri"]]
    original = path.stat()
    replacement = path.with_name("replacement")
    replacement.write_bytes(path.read_bytes())
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    replacement.replace(path)
    with pytest.raises(ValueError, match="snapshot changed"):
        reader.sample(0)
