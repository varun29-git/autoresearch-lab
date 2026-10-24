"""Generate an external AMD test config without launching hardware."""
import argparse
from dataclasses import asdict
import json

from production.artifacts import external_path
from production.build_dense_config import build_config as baseline_config
from terminus.data import TokenManifest
from terminus.integrity import write_json_atomic
from terminus.train import TrainConfig, load_config
from .recipe import CANDIDATES, changes, model_for


def build_config(*, manifest, world_size, global_batch_tokens, warmup_tokens, decay_tokens,
                 candidate="fp8", attention="rocm_aiter", local_attention="", wgrad_splits=None,
                 loss_chunk_size=4096, ddp_bucket_cap_mb=25):
    config = baseline_config(manifest=manifest, world_size=world_size, global_batch_tokens=global_batch_tokens,
                             warmup_tokens=warmup_tokens, decay_tokens=decay_tokens)
    config["model"] = asdict(model_for(candidate, attention=attention, local_attention=local_attention,
                                      wgrad_splits=wgrad_splits, loss_chunk_size=loss_chunk_size))
    config["training"]["ddp_bucket_cap_mb"] = ddp_bucket_cap_mb
    # Stability probes add full tensor scans. Keep them available and explicit;
    # performance measurements record whether they were enabled.
    TrainConfig(**config["training"])
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, required=True)
    for name in ("global-batch-tokens", "warmup-tokens", "decay-tokens"):
        p.add_argument("--" + name, type=int, required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--candidate", choices=CANDIDATES, default="fp8")
    p.add_argument("--attention", choices=("rocm_aiter", "rocm_ck"), default="rocm_aiter")
    p.add_argument("--local-attention", choices=("", "rocm_aiter", "rocm_ck"), default="")
    p.add_argument("--wgrad-splits", type=int, choices=(0, 1, 2, 4, 8, 16),
                   help="default: 4 for local FP8 wgrad; 0 for TE/BF16")
    p.add_argument("--loss-chunk-size", type=int, default=4096)
    p.add_argument("--ddp-bucket-cap-mb", type=int, default=25)
    args = p.parse_args()
    output = external_path(args.output, fresh=True)
    options = vars(args).copy()
    options.pop("output")
    options["manifest"] = TokenManifest.load(external_path(args.manifest))
    config = build_config(**options)
    write_json_atomic(output, config)
    model, _, _ = load_config(output)
    print(json.dumps({"output": str(output), "candidate": args.candidate, "changes": changes(model),
                      "world_size": args.world_size, "global_batch_tokens": args.global_batch_tokens,
                      "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
                      "status": "prepared_for_native_testing_not_certified"}, indent=2))


if __name__ == "__main__":
    main()
