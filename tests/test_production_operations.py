import copy
from dataclasses import asdict, replace
import fcntl
import json

import pytest
import torch

from production.artifacts import verified_checkpoint, verify_tokenizer
from production.checkpoints import audit, maintain
from production.evaluate import load_tasks
from production.export import export_checkpoint, load_export
from production.runtime import verify_runtime
from production.status import status
from production.qat_template import template
from production.compare_eval import compare
from production.evaluate import evaluate_checkpoint
from terminus.checkpoint import rng_state, save_checkpoint
from terminus.integrity import file_sha256, fingerprint, write_json_atomic
from terminus.model import BitNetLlama
from tests.helpers import tiny_config


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    tokenizer = tmp_path / "tokenizer.txt"
    tokenizer.write_text("synthetic identity fixture; no corpus")
    model = BitNetLlama(tiny_config(value_residual="learned", branch_norm="peri")).eval()
    contract = {"model": asdict(model.config), "data": {"tokenizer": {"sha256": file_sha256(tokenizer)}}}
    write_json_atomic(root / "run.json", {"contract": contract, "contract_sha256": fingerprint(contract)})
    optimizer = torch.optim.AdamW(model.parameters())
    for step in range(1, 4):
        save_checkpoint(root / "checkpoints", model, optimizer,
                        {"updates": step, "tokens_seen": step * 16, "next_micro_step": step},
                        contract, [rng_state()])
    return root, tokenizer, model


def test_checkpoint_identity_includes_attention_mask(run):
    root, tokenizer, model = run
    payload, _ = verified_checkpoint(root / "checkpoints", model_config=model.config)
    verify_tokenizer(tokenizer, payload["contract"])
    with pytest.raises(ValueError, match="differs"):
        verified_checkpoint(root / "checkpoints", model_config=replace(model.config, local_attention_window=2))
    tokenizer.write_text("changed")
    with pytest.raises(ValueError, match="tokenizer"):
        verify_tokenizer(tokenizer, payload["contract"])


def test_retention_copies_then_prunes_and_archive_can_recover(run, tmp_path):
    root, _, _ = run
    archive = tmp_path / "archive"
    plan = maintain(root, archive=archive, keep_last=1)
    assert plan["prune"] == ["step_00000001.pt", "step_00000002.pt"]
    assert not archive.exists()
    maintain(root, archive=archive, keep_last=1, protect=["step_00000001.pt"], apply=True)
    assert len(audit(root)["valid"]) == 2
    assert len(audit(archive)["valid"]) == 3
    payload, _ = verified_checkpoint(archive / "checkpoints")
    assert payload["state"]["updates"] == 3


def test_retention_refuses_active_run_and_corruption(run, tmp_path):
    root, _, _ = run
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            maintain(root, archive=tmp_path / "archive", apply=True)
    (root / "checkpoints/step_00000002.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="invalid"):
        maintain(root, archive=tmp_path / "archive", apply=True)
    assert len(list((root / "checkpoints").glob("*.pt"))) == 3
    assert audit(root)["recovery_checkpoint"] == "step_00000003.pt"


def test_export_roundtrip_and_tamper_rejection(run, tmp_path):
    root, tokenizer, model = run
    output = tmp_path / "export"
    receipt = export_checkpoint(root / "checkpoints", tokenizer, output)
    assert receipt["matrix_storage_bits"] == 2
    packed, copied_tokenizer, _ = load_export(output)
    assert copied_tokenizer.read_bytes() == tokenizer.read_bytes()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(packed(ids)["logits"], model(ids)["logits"], atol=2e-5, rtol=2e-5)
    with pytest.raises(FileExistsError):
        export_checkpoint(root / "checkpoints", tokenizer, output)
    copied_tokenizer.write_text("corrupted")
    with pytest.raises(ValueError, match="digest"):
        load_export(output)


def test_required_task_hash_version_and_final_role(tmp_path):
    task = tmp_path / "items.jsonl"
    task.write_text(json.dumps({"input": "prompt", "output": "answer"}) + "\n")
    manifest = tmp_path / "eval.json"
    manifest.write_text(json.dumps({"tasks": {"toy": {"status": "fetched", "revision": "v1",
        "file": task.name, "sha256": file_sha256(task), "role": "final"}}}))
    with pytest.raises(ValueError, match="final"):
        load_tasks(manifest, tmp_path, tasks=["toy"])
    items, files = load_tasks(manifest, tmp_path, tasks=["toy"], include_final=True)
    assert len(items) == 1 and files["toy"]["version"] == "v1"
    with pytest.raises(ValueError, match="unavailable"):
        load_tasks(manifest, tmp_path, tasks=["missing"])
    task.write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        load_tasks(manifest, tmp_path, tasks=["toy"], include_final=True)


def test_runtime_digest_and_compiler_drift():
    identity = {"compilers": {"deepgemm_jit": "a"}, "runtime": {"torch": "2.8"}}
    report = {"schema": 1, "identity": identity, "identity_sha256": fingerprint(identity)}
    verify_runtime(report, report)
    changed = copy.deepcopy(report)
    changed["identity"]["compilers"]["deepgemm_jit"] = "b"
    changed["identity_sha256"] = fingerprint(changed["identity"])
    with pytest.raises(ValueError, match="compilers"):
        verify_runtime(report, changed)


def test_evaluation_and_cpu_export_comparison(run, tmp_path, monkeypatch):
    from tests.test_task_eval import ToyTokenizer
    monkeypatch.setattr("production.evaluate.load_tokenizer", lambda path: ToyTokenizer())
    root, tokenizer, _ = run
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({"id": "one", "input": "abc", "output": "de", "language": "hi"}) + "\n")
    manifest = tmp_path / "evaluation.json"
    manifest.write_text(json.dumps({"tasks": {"toy": {"file": items.name, "status": "fetched",
                        "sha256": file_sha256(items), "version": "1", "role": "development"}}}))
    before = evaluate_checkpoint(root / "checkpoints", tokenizer, manifest, tmp_path,
                                 tmp_path / "before.json", tasks=["toy"])
    export_checkpoint(root / "checkpoints", tokenizer, tmp_path / "cpu")
    after = evaluate_checkpoint(None, None, manifest, tmp_path, tmp_path / "after.json",
                                tasks=["toy"], bundle=tmp_path / "cpu")
    report = compare(before, after, max_loss_increase=.001)
    assert report["status"] == "passed" and report["per_language"]["hi"]["n"] == 1
    assert compare(before, after)["status"] == "comparison_only"
    changed = copy.deepcopy(after)
    changed["tokenizer_sha256"] = "different"
    with pytest.raises(ValueError, match="tokenizer"):
        compare(before, changed)


def test_qat_values_remain_unset():
    spec = template()
    assert spec["status"] == "incomplete_not_a_training_config"
    assert spec["dense_tokens"] is None and spec["qat_tokens"] is None
    assert all(spec["optimizer"][key] is None for key in ("muon_lr", "adamw_lr", "norm_lr", "qk_norm_lr"))
    assert spec["optimizer"]["muon_lr_rule"] == "spectral_unclamped"
    assert spec["model"]["n_layers"] == 24


def test_health_reports_staleness_and_incidents(tmp_path):
    (tmp_path / "events.jsonl").write_text(json.dumps({"event": "train", "time_unix": 10}) + "\n")
    result = status(tmp_path, min_free_bytes=0, max_idle_seconds=5, now=20)
    assert result["alerts"] == ["training_log_stale"]
    (tmp_path / "events.jsonl").write_text(json.dumps({"event": "stopped", "time_unix": 10}) + "\n")
    assert not status(tmp_path, min_free_bytes=0, max_idle_seconds=5, now=20)["alerts"]
    (tmp_path / "incidents.jsonl").write_text(json.dumps({"event": "run_failure", "time_unix": 15}) + "\n")
    assert status(tmp_path, min_free_bytes=0, max_idle_seconds=5, now=20)["alerts"] == ["unresolved_incident"]
