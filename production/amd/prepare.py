"""Generate one selected MI325X profile outside Git; never launches training."""
import argparse
from dataclasses import asdict
import json
import shlex

from production.artifacts import external_path
from terminus.data import TokenManifest
from terminus.integrity import file_sha256, write_json_atomic
from terminus.train import load_config
from .profiles import PROFILES, select
from .recipe import CANDIDATES


def prepare(*, manifest_path, output_dir, profile="balanced", purpose="training", candidate=None):
    source = external_path(manifest_path)
    output = external_path(output_dir, fresh=True)
    derived, config, report = select(TokenManifest.load(source), profile=profile, purpose=purpose,
                                     candidate=candidate)
    report["input_manifest_file_sha256"] = file_sha256(source)
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (("manifest.json", asdict(derived)), ("config.json", config), ("selection.json", report)):
        write_json_atomic(output / name, payload)
    load_config(output / "config.json")
    chosen = PROFILES[profile]
    environment = {"CONFIG": str(output / "config.json"), "MANIFEST": str(output / "manifest.json"),
                   "OUTPUT_DIR": str(output / "run"), "PREFLIGHT_REPORT": str(output / "preflight.json"),
                   "RUNTIME_REPORT": str(output / "runtime.json"), "STAGING_RECEIPT": str(output / "staging.json"),
                   "EXPECTED_WORLD_SIZE": str(chosen.world_size), "NPROC_PER_NODE": str(chosen.world_size),
                   "EXPECTED_GLOBAL_BATCH_TOKENS": str(chosen.global_batch_tokens),
                   "NNODES": "1", "NODE_RANK": "0", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}
    (output / "env.sh").write_text("# Prepared only. Set DATA_ROOT; create/pass all receipts before launch.\n" +
                                  "".join(f"export {key}={shlex.quote(value)}\n" for key, value in environment.items()))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True, help="fresh directory outside Git")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="balanced")
    parser.add_argument("--purpose", choices=("training", "smoke"), default="training")
    parser.add_argument("--candidate", choices=CANDIDATES,
                        help="explicit backend override, e.g. --profile mb8 --candidate fp8")
    args = parser.parse_args()
    report = prepare(manifest_path=args.manifest, output_dir=args.output_dir,
                     profile=args.profile, purpose=args.purpose, candidate=args.candidate)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
