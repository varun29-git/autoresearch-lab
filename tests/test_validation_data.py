import json

import numpy as np
import pytest

from production.validation_data import (
    Reader, build, family_id, filter_index, indexed_span, normalized_weights,
    require_training_binding, role_for, verify, weighted_summary,
)


def make_bundle(tmp_path, *, missing=False):
    seed = "heldout-family-test"
    rows = []
    values = []
    for role in ("development", "final"):
        i = 0
        while role_for(family_id({"uuid": f"{role}-{i}"}), seed) != role:
            i += 1
        rows.append({"uuid": f"{role}-{i}", "token_start": len(values), "token_count": 18})
        values.extend(range(100 + len(values), 118 + len(values)))
    bin_path = tmp_path / "content.uint16"
    np.array(values, dtype="<u2").tofile(bin_path)
    cells = [{"id": lang + "/web", "language": lang, "parent": "web",
              "frames": [{"tokens_uri": str(bin_path), "documents": rows}]} for lang in ("en", "hi")]
    if missing:
        cells.append({"id": "en/missing", "language": "en", "parent": "missing", "frames": []})
    catalog, policy = tmp_path / "catalog.json", tmp_path / "policy.json"
    catalog.write_text(json.dumps({"cells": cells}))
    policy.write_text(json.dumps({"schema": "terminus_indexed_validation_v1", "seed": seed,
        "sequence_length": 8, "target_sequences_per_cell": 2, "minimum_sequences_per_cell": 1}))
    root = tmp_path / "bundle"
    return root, build(catalog, policy, root), rows, bin_path


def test_family_roles_and_original_content(tmp_path):
    root, receipt, rows, _ = make_bundle(tmp_path)
    assert verify(root) == receipt
    assert not receipt["missing_cells"]
    exclusions = [json.loads(x) for x in (root / "training-exclusions.jsonl").read_text().splitlines()]
    assert len(exclusions) == 2
    assert all(len(r["locations"]) == 2 for r in exclusions)
    assert {r["role"] for r in exclusions} == {"development", "final"}
    for name, meta in receipt["files"].items():
        ids = np.fromfile(root / name, dtype="<u2")
        start = 100 if meta["role"] == "development" else 118
        assert ids.tolist() == list(range(start, start + 17))
        assert meta["num_tokens"] == 2 * 8 + 1
    assert family_id({"uuid": "same"}) == family_id({"doc_id": "same"})
    with pytest.raises(ValueError, match="stable identity"):
        family_id({})


def test_whole_document_end_marker(tmp_path):
    root, _, _, _ = make_bundle(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy = json.loads(policy_path.read_text())
    policy.update(sequence_length=9, target_sequences_per_cell=2)
    policy_path.write_text(json.dumps(policy))
    other = tmp_path / "other"
    report = build(tmp_path / "catalog.json", policy_path, other)
    assert len(report["files"]) == 4
    for name in report["files"]:
        ids = np.fromfile(other / name, dtype="<u2")
        assert len(ids) == 19 and ids[-1] == 4
        assert np.all(ids[:-1] >= 32)


def test_holdout_filter_reserves_complete_family_across_languages(tmp_path):
    root, report, rows, _ = make_bundle(tmp_path)
    source = tmp_path / "training.index.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows + [
        {"uuid": "fresh-train", "token_start": 36, "token_count": 22}]) + "\n")
    output = tmp_path / "retained.jsonl"
    result = filter_index(root, source, output)
    assert result["removed_documents"] == 2 and result["retained_content_tokens"] == 22
    assert json.loads(output.read_text())["uuid"] == "fresh-train"
    assert not result["packed_training_verified"]
    training = tmp_path / "training-manifest.json"
    binding = {"manifest_sha256": report["manifest_sha256"],
               "exclusions_sha256": report["exclusions_sha256"], "mode": "loss_masked"}
    training.write_text(json.dumps({"provenance": {"validation_exclusion": binding}}))
    with pytest.raises(ValueError, match="document exclusions"):
        require_training_binding(root, training)


def test_missing_cells_cannot_pass_binding_or_weighted_score(tmp_path):
    root, report, _, _ = make_bundle(tmp_path, missing=True)
    assert report["status"] == "coverage_incomplete" and "en/missing" in report["missing_cells"]
    training = tmp_path / "manifest.json"
    training.write_text(json.dumps({"provenance": {"validation_exclusion": {
        "manifest_sha256": report["manifest_sha256"], "exclusions_sha256": report["exclusions_sha256"],
        "mode": "documents_excluded_before_packing"}}}))
    with pytest.raises(ValueError):
        require_training_binding(root, training)
    with pytest.raises(ValueError, match="coverage"):
        weighted_summary({"en/web": {"tokens": 10, "loss": 2.}}, {"en/web": .5, "en/missing": .5})


def test_digest_and_source_mutation(tmp_path):
    root, report, _, source = make_bundle(tmp_path)
    p = root / next(iter(report["files"]))
    p.write_bytes(p.read_bytes()[:-2])
    with pytest.raises(ValueError, match="identity"):
        verify(root)
    reader = Reader()
    reader.read(str(source), start=0, count=2)
    source.write_bytes(b"\x00\x00" + source.read_bytes()[2:])
    with pytest.raises(ValueError, match="source changed"):
        reader.read(str(source), start=0, count=2)
    with pytest.raises(ValueError, match="exceeds"):
        Reader().read(str(source), start=source.stat().st_size, count=2)


def test_manifest_cannot_claim_coverage_without_a_shard(tmp_path):
    from terminus.integrity import fingerprint
    root, report, _, _ = make_bundle(tmp_path)
    report["files"].pop(next(iter(report["files"])))
    report["manifest_sha256"] = fingerprint({k: v for k, v in report.items() if k != "manifest_sha256"})
    (root / "manifest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="coverage"):
        verify(root)


@pytest.mark.parametrize("row", [{"token_start": -1, "n_tokens": 2},
    {"token_start": 0, "n_tokens": 0}, {"token_start": True, "n_tokens": 2}])
def test_reject_invalid_spans(row):
    with pytest.raises(ValueError):
        indexed_span(row)


def test_exact_contract_weights_and_no_invalid_nll():
    budgets = {"en": 500_000_000_000, "hi": 100_000_000_000,
               "bn": 66_666_666_666, "gu": 66_666_666_666,
               **{k: 66_666_666_667 for k in ("ta", "te", "kn", "mr")}}
    weights = normalized_weights(budgets)
    assert sum(budgets.values()) == 1_000_000_000_000
    assert weights["en"] == .5 and weights["hi"] == .1
    rows = {k: {"loss": float(i + 1), "tokens": 4096} for i, k in enumerate(weights)}
    assert weighted_summary(rows, weights)["weighted_nll"] == sum(weights[k] * rows[k]["loss"] for k in weights)
    rows["hi"]["loss"] = float("nan")
    with pytest.raises(ValueError, match="record"):
        weighted_summary(rows, weights)
