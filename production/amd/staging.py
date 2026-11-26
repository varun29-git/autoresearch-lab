"""Full CPU verification once per immutable node-local snapshot.

Fast startup rechecks every file's resolved path, inode, size, mode, mtime and
ctime. This is an accidental-change guard, not a hostile-storage attestation.
Keep shards/tokenizer read-only for the run; changing any file invalidates it.
"""
import argparse
from dataclasses import asdict, replace
import copy
import json
import os
from pathlib import Path
import stat

from production.artifacts import external_path
from terminus.data import TokenManifest
from terminus.integrity import fingerprint, source_hashes, validate_manifest, write_json_atomic
from terminus.train import load_config


def snapshot(manifest, root):
    root = Path(root).resolve()
    names = [s.path for s in manifest.shards]
    if manifest.is_packed:
        from terminus.packed_data import stream_spec

        names += list(stream_spec(manifest)["paths"].values())
        names = sorted(set(names))
    if manifest.tokenizer:
        names.append(manifest.tokenizer["path"])
    result = {}
    for name in names:
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError("staged files must resolve inside the data root")
        st = path.stat()
        readonly_mount = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
        if not stat.S_ISREG(st.st_mode) or (st.st_mode & 0o222 and not readonly_mount):
            raise ValueError(f"staging requires read-only regular files: {name}")
        result[str(name)] = {"path": str(path), "device": st.st_dev, "inode": st.st_ino,
                             "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                             "ctime_ns": st.st_ctime_ns, "mode": st.st_mode,
                             "read_only_filesystem": readonly_mount}
    return result


def create(manifest, root, model, *, development_mode=False):
    root = Path(root).resolve()
    before = snapshot(manifest, root)
    validation = validate_manifest(manifest, root, model, development_mode=development_mode)
    after = snapshot(manifest, root)
    if before != after:
        raise ValueError("dataset changed during full verification")
    result = {"schema": 1, "kind": "ci1_amd_staging", "root": str(root), "files": after,
              "manifest_sha256": fingerprint(asdict(manifest)), "model_sha256": fingerprint(asdict(model)),
              "development_mode": development_mode, "source": source_hashes(), "validation": validation}
    result["receipt_sha256"] = fingerprint(result)
    return result


def verify(receipt, manifest, root, model, *, development_mode=False):
    expected = {"schema": 1, "kind": "ci1_amd_staging", "root": str(Path(root).resolve()),
                "manifest_sha256": fingerprint(asdict(manifest)), "model_sha256": fingerprint(asdict(model)),
                "development_mode": development_mode, "source": source_hashes()}
    if (any(receipt.get(k) != v for k, v in expected.items())
            or receipt.get("receipt_sha256") != fingerprint({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or not receipt.get("validation") or receipt.get("files") != snapshot(manifest, root)):
        raise ValueError("staged data receipt is stale or files changed; fully verify this snapshot again")
    return receipt["validation"]


def rebind_microbatch(receipt, original, derived, root, model, *, development_mode=False):
    """Reuse full packed acceptance only when batch_size is the sole change."""
    if not original.is_packed or not derived.is_packed:
        raise ValueError("microbatch receipt reuse requires packed manifests")
    expected = replace(original, config=replace(original.config, batch_size=derived.config.batch_size))
    if expected != derived:
        raise ValueError("receipt reuse permits only a packed microbatch change")
    verify(receipt, original, root, model, development_mode=development_mode)
    result = copy.deepcopy(receipt)
    result["manifest_sha256"] = fingerprint(asdict(derived))
    result["validation"]["manifest_sha256"] = result["manifest_sha256"]
    result["validation_reused_from_receipt_sha256"] = receipt["receipt_sha256"]
    result.pop("receipt_sha256")
    result["receipt_sha256"] = fingerprint(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "data-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--rebind-from", help="Reuse a verified packed receipt for a microbatch-only change")
    p.add_argument("--original-manifest", help="Exact manifest belonging to --rebind-from")
    args = p.parse_args()
    if bool(args.rebind_from) != bool(args.original_manifest):
        p.error("receipt reuse requires both --rebind-from and --original-manifest")
    model, training, _ = load_config(external_path(args.config))
    manifest = TokenManifest.load(external_path(args.manifest))
    if args.rebind_from:
        result = rebind_microbatch(json.loads(external_path(args.rebind_from).read_text()),
            TokenManifest.load(external_path(args.original_manifest)), manifest,
            external_path(args.data_root), model, development_mode=training.development_mode)
    else:
        result = create(manifest, external_path(args.data_root), model, development_mode=training.development_mode)
    write_json_atomic(external_path(args.output, fresh=True), result)
    print(json.dumps({"status": "snapshot_verified", "files": len(result["files"]),
                      "receipt_sha256": result["receipt_sha256"]}))


if __name__ == "__main__":
    main()
