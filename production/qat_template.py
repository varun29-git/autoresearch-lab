"""Write an intentionally incomplete QAT specification outside Git."""
import argparse

from production.artifacts import external_path
from production.recipe import SELECTED_MODEL, SELECTED_OPTIMIZER, SELECTED_TRAINING
from terminus.integrity import write_json_atomic


def template():
    optimizer = SELECTED_OPTIMIZER.copy()
    for key in ("muon_lr", "adamw_lr", "norm_lr", "qk_norm_lr"):
        optimizer[key] = None
    return {"schema": 1, "status": "incomplete_not_a_training_config",
            "dense_tokens": None, "qat_tokens": None, "global_batch_tokens": None,
            "model": {**SELECTED_MODEL, "weight_mode": "ternary"}, "optimizer": optimizer,
            "training": {**SELECTED_TRAINING, "warmup_tokens": None, "decay_tokens": None,
                         "gradient_accumulation_steps": None, "phase_start_tokens": None},
            "data_policy": None, "intentional_resets": None,
            "quality_max_loss_increase": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json_atomic(external_path(args.output, fresh=True), template())


if __name__ == "__main__":
    main()
