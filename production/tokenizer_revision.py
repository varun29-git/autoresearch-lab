"""Build the reviewed 65K atom-coverage candidate without resizing vocabulary.

Only twelve retired placeholders change. General reserved IDs 5-31, every
previously emitted content ID, English merges, routing and normalization remain
in place. Generated artifacts and qualification receipts stay outside Git.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from terminus import unicode_contract as unicodedata
from terminus.hashing import file_sha256
from terminus.tokenizer65k import Frozen65kTokenizer, TOKENIZER_SHA256

ATOM_CANDIDATE_SHA256 = "dc636d96224d059d03619e61f0ecddc3ac1e5c8c6fb46ee051f66cabc65fb35c"
ATOM_REPLACEMENTS = (
    (545, "\u200c"), (551, "ಠ"), (654, "ಃ"), (1137, "ऋ"),
    (1370, "ॲ"), (1697, "ঐ"), (2777, "ঢ"), (4233, "ઞ"),
    (4542, "ஊ"), (7859, "ఊ"), (9042, "ఠ"), (39992, "ಊ"),
)


def build_candidate(base):
    if file_sha256(base) != TOKENIZER_SHA256:
        raise ValueError("atom revision requires the exact original 65K artifact")
    artifact = json.loads(Path(base).read_text(encoding="utf-8"))
    score = min(s for s in artifact["indic_scores"] if s is not None) - 1
    changes = []
    for token_id, char in ATOM_REPLACEMENTS:
        old = artifact["pieces"][token_id]
        if old != f"<reserved_space_{token_id}>" or artifact["indic_scores"][token_id] is not None:
            raise ValueError("revision would overwrite an active vocabulary ID")
        if char in artifact["pieces"] or unicodedata.normalize("NFC", char) != char:
            raise ValueError("new atom is duplicate or noncanonical")
        changes.append({"id": token_id, "before": old, "after": char,
                        "unicode_name": unicodedata.name(char), "old_score": None,
                        "new_score": score})
        artifact["pieces"][token_id] = char
        artifact["indic_scores"][token_id] = score
    artifact["production_revision"] = {
        "version": 2, "base_artifact_sha256": TOKENIZER_SHA256,
        "change": "replace twelve retired space placeholders with Indic single-character atoms; retain reserved IDs 5-31",
        "changed_ids": [r["id"] for r in changes],
    }
    object.__new__(Frozen65kTokenizer)._validate_artifact(artifact)
    payload = (json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if hashlib.sha256(payload).hexdigest() != ATOM_CANDIDATE_SHA256:
        raise ValueError("candidate artifact differs from the reviewed patch")
    return payload, changes


def expand_atom_ids(ids):
    """Qualification oracle: expand new atoms to their original fallback IDs."""
    expansions = {token: [288 + b for b in char.encode("utf-8")]
                  for token, char in ATOM_REPLACEMENTS}
    return [old for token in ids for old in expansions.get(token, [token])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    output = Path(args.output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        raise ValueError("generated tokenizer artifacts must be outside the repository")
    payload, changes = build_candidate(args.base)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps({"artifact": str(output), "sha256": ATOM_CANDIDATE_SHA256,
                      "changed_ids": [r["id"] for r in changes], "reserved_ids": list(range(5, 32)),
                      "production_promotion_requires_qualification": True}))


if __name__ == "__main__":
    main()
