"""Seal indexed, already-tokenized documents into independent evaluation roles.

Catalogs, policies, documents, exclusions and generated shards stay outside Git.
The catalog defines the sampling frame; this tool does not certify population
representativeness or benchmark contamination. It never re-encodes token IDs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
from urllib.parse import urlparse

import numpy as np

from production.artifacts import external_path
from terminus.integrity import file_sha256, fingerprint, write_json_atomic

ROLES = ("development", "final")
SCHEMA = "terminus_indexed_validation_v1"


def family_id(row):
    explicit = row.get("family_id")
    if explicit:
        identity = ["explicit", str(explicit)]
    else:
        value = next((row[k] for k in ("doc_id", "uuid", "document_id", "id")
                      if row.get(k) is not None), None)
        if value is None:
            if not row.get("row_key"):
                raise ValueError("document lacks stable identity")
            identity = ["row", str(row["row_key"])]
        else:
            # Equal published IDs are conservatively grouped across languages
            # and sources. This can overexclude; it does not resolve unknown
            # translation families with different IDs.
            identity = ["published_id", str(value)]
    return fingerprint(identity)


def role_for(family, seed):
    value = hashlib.sha256((seed + ":role:" + family).encode()).digest()[0]
    return ROLES[value % 2]


def indexed_span(row):
    start = row.get("token_start")
    count = row.get("n_tokens", row.get("token_count"))
    if (type(start) is not int or type(count) is not int or start < 0 or count <= 0):
        raise ValueError("invalid document token span")
    return start, count


def normalized_weights(budgets):
    if not budgets or any(type(v) is not int or v <= 0 for v in budgets.values()):
        raise ValueError("weights need positive integer token budgets")
    total = sum(budgets.values())
    return {k: v / total for k, v in budgets.items()}


def weighted_summary(records, weights):
    if not weights or any(not np.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("invalid evaluation weights")
    if abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("evaluation weights must sum to one")
    if set(records) != set(weights):
        raise ValueError("evaluation coverage differs from the frozen contract")
    for row in records.values():
        if (type(row.get("tokens")) is not int or row["tokens"] <= 0
                or not np.isfinite(row.get("loss", float("nan")))):
            raise ValueError("invalid evaluation record")
    return {"weighted_nll": sum(weights[k] * records[k]["loss"] for k in weights),
            "macro_nll": sum(r["loss"] for r in records.values()) / len(records),
            "per_cell": records}


class Reader:
    def __init__(self):
        self.client = None
        self.identities = {}

    def read(self, uri, *, start=None, count=None):
        if start is not None and (type(start) is not int or start < 0 or
                                  type(count) is not int or count <= 0):
            raise ValueError("invalid byte range")
        if uri.startswith("s3://"):
            import boto3
            from botocore.config import Config
            if self.client is None:
                self.client = boto3.client("s3", region_name="us-east-1", config=Config(
                    connect_timeout=10, read_timeout=60,
                    retries={"max_attempts": 2, "mode": "standard"}))
            u = urlparse(uri)
            if uri not in self.identities:
                head = self.client.head_object(Bucket=u.netloc, Key=u.path.lstrip("/"))
                self.identities[uri] = {"size_bytes": head["ContentLength"],
                                        "etag": head["ETag"],
                                        "version_id": head.get("VersionId")}
            identity = self.identities[uri]
            args = {"Bucket": u.netloc, "Key": u.path.lstrip("/"),
                    "IfMatch": identity["etag"]}
            if identity["version_id"]:
                args["VersionId"] = identity["version_id"]
            if start is not None:
                if start + count > identity["size_bytes"]:
                    raise ValueError("token span exceeds source object")
                args["Range"] = f"bytes={start}-{start + count - 1}"
            with self.client.get_object(**args)["Body"] as body:
                data = body.read()
        else:
            path = Path(uri)
            identity = {"size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path)}
            if uri in self.identities and self.identities[uri] != identity:
                raise ValueError("source changed during validation construction")
            self.identities[uri] = identity
            with path.open("rb") as handle:
                if start is not None:
                    if start + count > identity["size_bytes"]:
                        raise ValueError("token span exceeds source file")
                    handle.seek(start)
                data = handle.read(count if count is not None else -1)
        if count is not None and len(data) != count:
            raise ValueError("short source read")
        return data


def build(catalog_path, policy_path, output_dir, *, reader=None):
    catalog_path, policy_path = external_path(catalog_path), external_path(policy_path)
    output = external_path(output_dir, fresh=True)
    output.mkdir(parents=True, exist_ok=False)
    catalog, policy = [json.loads(p.read_text()) for p in (catalog_path, policy_path)]
    if policy.get("schema") != SCHEMA or not policy.get("seed"):
        raise ValueError("invalid validation policy")
    if policy.get("eod_id", 4) != 4:
        raise ValueError("validation requires the reviewed EOD token 4")
    seq_len, target = policy["sequence_length"], policy["target_sequences_per_cell"]
    if type(seq_len) is not int or seq_len <= 0 or type(target) is not int or target <= 0:
        raise ValueError("invalid validation sequence policy")
    minimum = policy.get("minimum_sequences_per_cell", 1)
    if type(minimum) is not int or not 0 < minimum <= target:
        raise ValueError("invalid validation minimum")
    cells = catalog["cells"]
    if not cells or len({c["id"] for c in cells}) != len(cells):
        raise ValueError("empty or duplicate catalog cells")
    reader = reader or Reader()
    # Prefetch a bounded prefix of each sampling frame in parallel. Existing
    # holdouts have many tiny files; serial HEAD/GET would dominate their cost.
    # Each worker pins its own identities, merged before consuming any IDs.
    local = threading.local()
    def prefetch(frame):
        spans = [indexed_span(row) for row in frame["documents"]]
        first, last = min(s for s, _ in spans), max(s + n for s, n in spans)
        if last - first > policy.get("max_frame_tokens", 2_000_000):
            raise ValueError("sampling frame exceeds bounded read policy")
        if not hasattr(local, "reader"):
            local.reader = Reader()
        worker = local.reader
        key = (frame["tokens_uri"], 2 * first, 2 * (last - first))
        raw = worker.read(key[0], start=key[1], count=key[2])
        return key, raw, {key[0]: worker.identities[key[0]]}
    prefetched = {}
    if type(reader) is Reader:
        jobs = []
        for cell in cells:
            available = 0
            for frame in cell.get("frames", []):
                if not frame["documents"]:
                    continue
                spans = [indexed_span(row) for row in frame["documents"]]
                jobs.append(frame)
                available += max(s + n for s, n in spans) - min(s for s, _ in spans)
                if available >= 2.5 * target * seq_len + 2:
                    break
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, result in enumerate(pool.map(prefetch, jobs)):
                key, raw, identities = result
                for uri, identity in identities.items():
                    if uri in reader.identities and reader.identities[uri] != identity:
                        raise ValueError("source changed during parallel validation reads")
                    reader.identities[uri] = identity
                prefetched[key] = raw
                if (i + 1) % 100 == 0:
                    print(json.dumps({"event": "validation_prefetch", "frames": i + 1,
                                      "total_frames": len(jobs)}), flush=True)
    exclusions, coverage, files = {}, {}, {}
    seen = set()
    for cell in cells:
        streams = {role: [] for role in ROLES}
        sizes = {role: 0 for role in ROLES}
        documents = {role: 0 for role in ROLES}
        for frame in cell.get("frames", []):
            rows = frame["documents"]
            if not rows:
                continue
            spans = [indexed_span(row) for row in rows]
            first, last = min(s for s, _ in spans), max(s + n for s, n in spans)
            if last - first > policy.get("max_frame_tokens", 2_000_000):
                raise ValueError("sampling frame exceeds bounded read policy")
            key = (frame["tokens_uri"], 2 * first, 2 * (last - first))
            raw = prefetched.pop(key, None)
            if raw is None:
                raw = reader.read(key[0], start=key[1], count=key[2])
            values = np.frombuffer(raw, dtype="<u2")
            if "range_sha256" in frame and hashlib.sha256(raw).hexdigest() != frame["range_sha256"]:
                raise ValueError("sampling frame digest mismatch")
            frame["range_sha256"] = hashlib.sha256(raw).hexdigest()
            order = sorted(range(len(rows)), key=lambda i: fingerprint(
                [policy["seed"], cell["id"], family_id(rows[i])]))
            for i in order:
                row, (start, count) = rows[i], spans[i]
                family = family_id(row)
                role = role_for(family, policy["seed"])
                if sizes[role] >= target * seq_len + 1:
                    continue
                key = (cell["id"], family)
                if key in seen:
                    continue
                ids = values[start - first:start - first + count]
                if len(ids) != count or not len(ids) or ids.min() < 32:
                    raise ValueError("invalid content IDs or missing document span")
                seen.add(key)
                streams[role].extend((ids.tobytes(), np.array([4], dtype="<u2").tobytes()))
                sizes[role] += count + 1
                documents[role] += 1
                exclusions.setdefault(family, {"family_id": family, "role": role,
                    "locations": []})["locations"].append({"cell": cell["id"],
                        "tokens_uri": frame["tokens_uri"], "token_start": start,
                        "token_count": count, "doc_id": row.get("doc_id", row.get("uuid")),
                        "row_key": row.get("row_key"),
                        "preexisting_holdout": frame.get("preexisting_holdout", False)})
            if all(s >= target * seq_len + 1 for s in sizes.values()):
                break
        coverage[cell["id"]] = {}
        for role in ROLES:
            sequences = min(target, max(0, (sizes[role] - 1) // seq_len))
            coverage[cell["id"]][role] = {"sequences": sequences,
                "documents": documents[role], "language": cell["language"],
                "parent": cell["parent"], "available_stream_tokens": sizes[role]}
            if sequences:
                directory = output / role
                directory.mkdir(exist_ok=True)
                name = fingerprint(cell["id"])[:24] + ".uint16"
                data = b"".join(streams[role])[:2 * (sequences * seq_len + 1)]
                path = directory / name
                path.write_bytes(data)
                files[str(path.relative_to(output))] = {"cell": cell["id"], "role": role,
                    "language": cell["language"], "parent": cell["parent"],
                    "sha256": file_sha256(path), "num_tokens": len(data) // 2}
        print(json.dumps({"event": "validation_cell", "cell": cell["id"],
                          "coverage": coverage[cell["id"]]}), flush=True)
    exclusion_path = output / "training-exclusions.jsonl"
    exclusion_path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n"
                                      for _, v in sorted(exclusions.items())))
    missing = {k: v for k, v in coverage.items()
               if any(v[role]["sequences"] < minimum for role in ROLES)}
    receipt = {"schema": SCHEMA, "policy": policy,
        "catalog_sha256": file_sha256(catalog_path), "policy_sha256": file_sha256(policy_path),
        "coverage": coverage, "files": files, "missing_cells": missing,
        "tokenizer_lineages": catalog.get("tokenizer_lineages", []),
        "source_objects": reader.identities, "families": len(exclusions),
        "exclusions_sha256": file_sha256(exclusion_path),
        "unknown_translation_families_resolved": False,
        "benchmark_overlap_audited": False,
        "training_exclusions_applied": False,
        "status": "sealed_validation_ready_for_binding" if not missing else "coverage_incomplete"}
    receipt["manifest_sha256"] = fingerprint(receipt)
    write_json_atomic(output / "manifest.json", receipt)
    write_json_atomic(output / "sampling-frames.json", catalog)
    return receipt


def verify(bundle):
    root = external_path(bundle)
    report = json.loads((root / "manifest.json").read_text())
    if report.get("manifest_sha256") != fingerprint({k: v for k, v in report.items()
                                                    if k != "manifest_sha256"}):
        raise ValueError("validation manifest identity mismatch")
    actual_cells = set()
    length = report["policy"]["sequence_length"]
    for name, meta in report["files"].items():
        key = (meta["cell"], meta["role"])
        if key in actual_cells or meta["role"] not in ROLES:
            raise ValueError("duplicate or invalid validation cell role")
        actual_cells.add(key)
        sequences = report["coverage"][meta["cell"]][meta["role"]]["sequences"]
        if type(sequences) is not int or sequences <= 0 or meta["num_tokens"] != sequences * length + 1:
            raise ValueError("validation coverage length mismatch")
        path = (root / name).resolve()
        if root.resolve() not in path.parents or file_sha256(path) != meta["sha256"]:
            raise ValueError("validation file identity mismatch")
        if path.stat().st_size != 2 * meta["num_tokens"]:
            raise ValueError("validation file length mismatch")
    expected_cells = {(cell, role) for cell, entry in report["coverage"].items()
                      for role in ROLES if entry[role]["sequences"] > 0}
    if actual_cells != expected_cells:
        raise ValueError("validation coverage does not match sealed files")
    minimum = report["policy"].get("minimum_sequences_per_cell", 1)
    missing = {cell: entry for cell, entry in report["coverage"].items()
               if any(entry[r]["sequences"] < minimum for r in ROLES)}
    if missing != report["missing_cells"]:
        raise ValueError("validation missing-cell declaration mismatch")
    p = root / "training-exclusions.jsonl"
    if file_sha256(p) != report["exclusions_sha256"]:
        raise ValueError("exclusions identity mismatch")
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    if len({x["family_id"] for x in rows}) != len(rows):
        raise ValueError("validation family appears in multiple roles")
    if any(x["role"] not in ROLES for x in rows):
        raise ValueError("invalid evaluation role")
    if any(x["role"] != role_for(x["family_id"], report["policy"]["seed"]) for x in rows):
        raise ValueError("family role differs from the frozen split")
    return report


def filter_index(bundle, source_index, output_index):
    """Reserve known families before a document-aware packer reads their IDs.

    This filters an index, not the underlying token file. A packer must read
    exactly these retained document spans; concatenating the old bin unchanged
    would defeat the exclusion. The receipt deliberately does not certify
    packed training output.
    """
    report = verify(bundle)
    root = external_path(bundle)
    source, output = external_path(source_index), external_path(output_index, fresh=True)
    excluded = {json.loads(line)["family_id"] for line in
                (root / "training-exclusions.jsonl").read_text().splitlines()}
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = removed = tokens = 0
    with source.open() as src, output.open("x") as dst:
        for line in src:
            row = json.loads(line)
            _, count = indexed_span(row)
            if family_id(row) in excluded:
                removed += 1
            else:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                tokens += count
    receipt = {"manifest_sha256": report["manifest_sha256"],
        "exclusions_sha256": report["exclusions_sha256"],
        "source_index_sha256": file_sha256(source),
        "retained_index_sha256": file_sha256(output),
        "retained_documents": kept, "removed_documents": removed,
        "retained_content_tokens": tokens, "packed_training_verified": False}
    write_json_atomic(output.with_suffix(output.suffix + ".receipt.json"), receipt)
    return receipt


def require_training_binding(bundle, training_manifest):
    """Check declaration identities; packed exclusion execution needs an audit."""
    report = verify(bundle)
    training = json.loads(external_path(training_manifest).read_text())
    binding = training.get("provenance", {}).get("validation_exclusion", {})
    if (report["missing_cells"] or binding.get("manifest_sha256") != report["manifest_sha256"]
            or binding.get("exclusions_sha256") != report["exclusions_sha256"]
            or binding.get("mode") != "documents_excluded_before_packing"):
        raise ValueError("training corpus has not applied the frozen document exclusions")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for name in ("catalog", "policy", "output-dir"):
        b.add_argument("--" + name, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--bundle", required=True)
    v.add_argument("--training-manifest")
    f = sub.add_parser("filter-index")
    for name in ("bundle", "source-index", "output-index"):
        f.add_argument("--" + name, required=True)
    args = p.parse_args()
    if args.command == "build":
        report = build(args.catalog, args.policy, args.output_dir)
    elif args.command == "filter-index":
        print(json.dumps(filter_index(args.bundle, args.source_index, args.output_index)))
        return
    else:
        report = (require_training_binding(args.bundle, args.training_manifest)
                  if args.training_manifest else verify(args.bundle))
    print(json.dumps({"status": report["status"], "missing_cells": list(report["missing_cells"])}))


if __name__ == "__main__":
    main()
