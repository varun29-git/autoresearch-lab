"""Generate matched-shrinkage AdamW width-transfer candidates.

The hidden Muon rates and small AdamW groups are unchanged. Only the tied
embedding/readout AdamW rate follows the requested width exponent. Its decay
coefficient changes inversely so lr * weight_decay is held fixed per update.
This is a testable transfer hypothesis, not a full muP parameterization.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from .bridge import WIDTHS
from .loop import DEFAULT_REFERENCE, candidate_config, load_reference


EXPONENTS = (0.0, 0.5, 1.0)


def transfer_candidate(candidate: dict, reference: dict, *, width: int,
                       embedding_exponent: float) -> dict:
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {sorted(WIDTHS)}")
    if embedding_exponent not in EXPONENTS or not math.isfinite(embedding_exponent):
        raise ValueError("embedding exponent must be 0, 0.5, or 1")
    base_width = reference["config"]["model"]["dim"]
    if base_width != 768 or reference["config"]["optimizer"]["muon_lr_rule"] != "spectral_unclamped":
        raise ValueError("transfer needs the 768-width spectral-Muon reference")
    candidate_config(reference, candidate)
    result = copy.deepcopy(candidate)
    optimizer = result["optimizer"]
    lr_factor = (base_width / width) ** embedding_exponent
    optimizer["adamw_lr"] *= lr_factor
    optimizer["embedding_weight_decay"] /= lr_factor
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=sorted(WIDTHS), required=True)
    parser.add_argument("--embedding-exponent", type=float, choices=EXPONENTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"candidate already exists: {args.output}")
    candidate = json.loads(args.candidate.read_text())
    result = transfer_candidate(candidate, load_reference(args.reference),
                                width=args.width, embedding_exponent=args.embedding_exponent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "width": args.width,
                      "embedding_exponent": args.embedding_exponent,
                      "adamw_lr": result["optimizer"]["adamw_lr"],
                      "embedding_weight_decay": result["optimizer"]["embedding_weight_decay"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
