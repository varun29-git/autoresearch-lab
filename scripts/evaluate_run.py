"""Compatibility entry point for the production evaluation command.

Use --tasks scalingbench/bbh,scalingbench/math (manifest keys). Requested
missing tasks and implicit final-set access are rejected. See
production/OPERATIONS.md for the frozen evaluation contract.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from production.artifacts import verified_checkpoint
from production.evaluate import main
from terminus.model import BitNetLlama
from terminus.train import load_config


def load_verified_model(config_path, checkpoint_dir):
    model_config, _, _ = load_config(config_path)
    payload, receipt = verified_checkpoint(checkpoint_dir, model_config=model_config)
    model = BitNetLlama(model_config)
    model.load_state_dict(payload["model"])
    return model.eval(), model_config, payload["state"], receipt


if __name__ == "__main__":
    main()
