"""Create and verify a complete result-tree receipt, including logs/checkpoints.

Remote verification requires an independent download into a second directory.
Successful upload alone is not treated as proof that the archive is complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import sha256, write_json


def inventory(root: Path, *, exclude: Path | None = None) -> dict:
    root = root.resolve()
    excluded = exclude.resolve() if exclude else None
    if not root.is_dir():
        raise ValueError(f"result root does not exist: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"result archive cannot contain a symlink: {path}")
        if path.is_file() and path.resolve() != excluded:
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size,
                                                  "sha256": sha256(path)}
    return files


def create(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("receipt must live inside the result tree")
    files = inventory(root, exclude=output)
    receipt = {"version": 1, "files": files, "total_bytes": sum(x["bytes"] for x in files.values())}
    write_json(output, receipt)
    return {"files": len(files), "total_bytes": receipt["total_bytes"], "receipt": str(output)}


def verify(root: Path, receipt_path: Path) -> dict:
    root, receipt_path = root.resolve(), receipt_path.resolve()
    if receipt_path.is_relative_to(root):
        raise ValueError("verification needs the trusted local receipt outside the downloaded tree")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("version") != 1:
        raise ValueError("unsupported artifact receipt")
    expected = receipt["files"]
    downloaded_receipt = root / receipt_path.name
    if not downloaded_receipt.is_file() or sha256(downloaded_receipt) != sha256(receipt_path):
        raise ValueError("downloaded archive receipt differs from the trusted local receipt")
    actual = inventory(root, exclude=root / receipt_path.name)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path])
        raise ValueError(f"archive verification failed: missing={missing}, extra={extra}, changed={changed}")
    return {"verified": True, "files": len(actual),
            "total_bytes": sum(x["bytes"] for x in actual.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = create(args.root, args.receipt) if args.command == "create" else verify(args.root, args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
