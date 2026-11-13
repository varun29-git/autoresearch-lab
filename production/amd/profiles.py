"""Reviewed MI325X starting choices; these are not measured GPU winners.

See CONFIG_SELECTION.md for evidence, comparison order and acceptance gates.
No manifest data, optimizer rates or architecture are selected by this module.
"""
from dataclasses import asdict, dataclass, replace

from terminus.data import TokenManifest
from terminus.integrity import fingerprint, manifest_scope
from terminus.model import BitNetLlamaConfig
from terminus.optim import OptimizerConfig
from terminus.train import TrainConfig
from .build_config import build_config
from .recipe import CANDIDATES, validate_production


GLOBAL_BATCH_TOKENS = 2_097_152
# Cadence multipliers below are even, so milestones align across 393k, 1M,
# 2M, 3.146M and 4M-token batches. This quantum itself is not their LCM.
COMPARISON_QUANTUM = 6_291_456
WARMUP_TOKENS = 160 * COMPARISON_QUANTUM
BRIDGE_END_TOKENS = 640 * COMPARISON_QUANTUM
VALIDATION_SEQUENCES = 1024  # Per language/domain, globally, at every microbatch.
MIN_TRAINING_TOKENS = (10_000_000_000 // 4096) * 4096
CADENCE_TOKENS = {
    "checkpoint_interval": 320 * COMPARISON_QUANTUM,
    "validation_interval": 320 * COMPARISON_QUANTUM,
    "stability_log_interval": 32 * COMPARISON_QUANTUM,
    "log_interval": 4 * COMPARISON_QUANTUM,
}


@dataclass(frozen=True)
class Profile:
    reason: str
    microbatch: int = 16
    world_size: int = 8
    global_batch_tokens: int = GLOBAL_BATCH_TOKENS
    candidate: str = "fp8_te"
    local_attention: str = ""
    ddp_bucket_cap_mb: int = 25
    fused_fp8_producers: bool = True


PROFILES = {
    "balanced": Profile(
        "Selected starting point. Microbatch 32, accumulation 2, global batch 2097152. AttnNorm, FfnNorm and packed SwiGLU emit the FNUZ row sidecar from the rounded BF16 value. Fit and the producer preflight are unmeasured.",
        microbatch=32),
    "mb4": Profile("First memory-conservative full update; same global batch.", microbatch=4),
    "mb8": Profile("Memory fallback / execution comparison; same global batch.", microbatch=8),
    "mb32": Profile(
        "Same microbatch, producers, and global batch as balanced. Named alias; fit is unmeasured.",
        microbatch=32),
    "mb32_fused": Profile(
        "Eight-GPU continuation of world4_mb32. Microbatch 32, accumulation 2, same producers. Fit is unmeasured.",
        microbatch=32, fused_fp8_producers=True),
    "batch_1m_mb16": Profile("Intermediate global-batch quality candidate at microbatch 16.",
                            global_batch_tokens=1_048_576),
    "batch_3m_mb16": Profile("3.146M-token quality control at microbatch 16.",
                            global_batch_tokens=3_145_728),
    "batch_3m_mb24": Profile("Larger microbatch at the matched 3.146M global batch; HBM fit unmeasured.",
                            microbatch=24, global_batch_tokens=3_145_728),
    "aiter": Profile("Independent AITER forward/dgrad plus local FP32 wgrad.", candidate="fp8"),
    "triton": Profile("Independent local GEMM diagnostic, not a mandatory speed trial.", candidate="fp8_triton"),
    "bf16": Profile(
        "Quality control with projection quantization removed.",
        candidate="bf16", fused_fp8_producers=False),
    "batch_bridge": Profile("Historical global batch on the selected full model; quality control.",
                            microbatch=4, global_batch_tokens=393_216),
    "batch_bridge_mb12": Profile("Historical HPO global batch in one larger eight-GPU microstep; fit unmeasured.",
                                 microbatch=12, global_batch_tokens=393_216),
    "large_batch": Profile("Quality-affecting 4M batch trial; preserve all optimizer rates.",
                           global_batch_tokens=4_194_304),
    "ddp64": Profile("Conditional communication-overlap trial.", ddp_bucket_cap_mb=64),
    "ddp128": Profile("Conditional communication-overlap trial.", ddp_bucket_cap_mb=128),
    "ck_local": Profile("Independent CK for the twenty local layers.", local_attention="rocm_ck"),
    "fused": Profile(
        "Eight-GPU continuation of world4. Norm and SwiGLU emit the FNUZ sidecar the next GEMM consumes. "
        "The producer preflight must pass before the first step.",
        fused_fp8_producers=True),
    "single_gpu": Profile("Matched-global-batch rank-equivalence control, not a scaling claim.", world_size=1),
    "world4": Profile(
        "Four-GPU launch. Same global batch, accumulation 8, with the norm and SwiGLU FP8 producers on. "
        "A finished update widens onto the fused profile.",
        world_size=4, fused_fp8_producers=True),
    "world4_mb32": Profile(
        "Four-GPU HBM attempt at the same global batch. Microbatch 32, accumulation 4, producers on. "
        "Fit on 256 GB is unmeasured. A finished update widens onto mb32_fused.",
        microbatch=32, world_size=4, fused_fp8_producers=True),
}


def validation_coverage(manifest):
    """Match train.evaluate's domain grouping and full-sequence accounting."""
    scope = manifest_scope(manifest)
    domains = set(scope["domains"])
    required = set(scope["languages"]) | domains
    counts = {}
    for spec in manifest.shards_for(split=manifest.config.val_split):
        key = spec.source if spec.source in domains else spec.language
        if key not in required:
            raise ValueError(f"validation shard outside declared scope: {spec.path}")
        counts[key] = counts.get(key, 0) + spec.num_tokens
    blocks = {key: max(0, (counts.get(key, 0) - 1) // manifest.config.seq_len)
              for key in sorted(required)}
    short = {key: count for key, count in blocks.items() if count < VALIDATION_SEQUENCES}
    if short:
        raise ValueError(f"need {VALIDATION_SEQUENCES} full validation sequences per scope key: {short}")
    return blocks


def select(manifest, *, profile="balanced", purpose="training", candidate=None):
    """Return a new manifest, trainer config and selection record, without I/O.

    Only microbatch changes in the supplied manifest. Full content/provenance
    verification remains the responsibility of the existing staging command.
    Smoke schedules are deliberately not convergence comparisons or resumable
    prefixes of the selected long schedule.
    """
    if profile not in PROFILES or purpose not in {"training", "smoke"}:
        raise ValueError("unknown profile or purpose")
    if candidate is not None and candidate not in CANDIDATES:
        raise ValueError("unknown candidate override")
    if manifest.config.seq_len != 4096 or manifest.config.token_dtype != "uint16":
        raise ValueError("selected profiles require 4096-token uint16 shards")
    selected = PROFILES[profile]
    if candidate is not None:
        selected = replace(selected, candidate=candidate)
    derived = TokenManifest(replace(manifest.config, batch_size=selected.microbatch),
                            manifest.shards, manifest.tokenizer, manifest.provenance)
    coverage = validation_coverage(derived)
    total, batch = derived.config.usable_train_tokens, selected.global_batch_tokens
    if purpose == "training":
        if total < MIN_TRAINING_TOKENS:
            raise ValueError("training profile needs at least 10B usable tokens; use purpose=smoke for bring-up")
        warmup = WARMUP_TOKENS
        # Round the final 10% up to a full sequence, never change data horizon.
        decay = ((derived.config.total_examples + 9) // 10) * 4096
    else:
        if total < 26 * batch:
            raise ValueError("smoke needs at least 26 complete global updates in the manifest")
        warmup, decay = 2 * batch, batch
    config = build_config(manifest=derived, world_size=selected.world_size,
                          global_batch_tokens=batch, warmup_tokens=warmup, decay_tokens=decay,
                          candidate=selected.candidate, local_attention=selected.local_attention,
                          ddp_bucket_cap_mb=selected.ddp_bucket_cap_mb,
                          fused_fp8_producers=selected.fused_fp8_producers)
    # Freeze training defaults too. Optimizer settings come from the baseline.
    training = asdict(TrainConfig(**config["training"]))
    training.update({name: max(1, tokens // batch) for name, tokens in CADENCE_TOKENS.items()})
    per_validation_step = selected.microbatch * selected.world_size
    training.update(validation_batches_per_language=(VALIDATION_SEQUENCES + per_validation_step - 1) //
                    per_validation_step, validation_sequences_per_language=VALIDATION_SEQUENCES,
                    prefetch_batches=4)
    if purpose == "smoke":
        training.update(checkpoint_interval=10, validation_interval=10,
                        stability_log_interval=1, log_interval=1)
    config["training"] = training
    validate_production(BitNetLlamaConfig(**config["model"]), TrainConfig(**training),
                        OptimizerConfig(**config["optimizer"]))
    allocation_seconds = (5760 - 320) * 3600 * .9
    report = {
        "schema": 1, "status": "selected_for_native_testing_not_certified",
        "profile": profile, "purpose": purpose, "selection": asdict(selected),
        "candidate_override": candidate,
        "input_manifest_identity": fingerprint(asdict(manifest)),
        "derived_manifest_identity": fingerprint(asdict(derived)),
        "config_identity": fingerprint(config),
        "usable_train_tokens": total, "discarded_subsequence_tail_tokens": manifest.config.total_train_tokens - total,
        "global_sequences_per_update": batch // 4096,
        "tokens_per_rank_per_update": batch // selected.world_size,
        "accumulation": training["gradient_accumulation_steps"],
        "optimizer_updates": (total + batch - 1) // batch, "last_update_tokens": total % batch or batch,
        "warmup_tokens": warmup, "decay_tokens": decay, "decay_start_tokens": total - decay,
        "cadence_tokens": {key: training[key] * batch for key in CADENCE_TOKENS},
        "validation_sequences_per_scope_key": VALIDATION_SEQUENCES,
        "validation_available_sequences": coverage,
        "validation_tokens_per_check": len(coverage) * VALIDATION_SEQUENCES * 4096,
        "bridge_stop_after_updates": BRIDGE_END_TOKENS // batch if purpose == "training" else None,
        "bridge_end_tokens": BRIDGE_END_TOKENS if purpose == "training" else None,
        "allocation": (
            {"scope": "four_gpu_prefix_of_the_eight_gpu_run", "gpus": 4, "continues_on_gpus": 8,
             "global_batch_tokens": batch, "accumulation": training["gradient_accumulation_steps"],
             "fp8_producers": True,
             "eight_gpu_profile": "fused" if profile == "world4" else "mb32_fused",
             "measured_rate": None,
             "note": "Same 2097152-token update and the same norm and SwiGLU FP8 producers as balanced. Widen a finished checkpoint and launch the named eight-GPU profile so the model contract matches. This is not the 5760 GPU-hour month."}
            if profile in {"world4", "world4_mb32"} else
            {"scope": "standalone_dense_only", "gpu_hours": 5760, "gpus": 8, "reserve_gpu_hours": 320,
             "assumed_production_duty_cycle": .9,
             "required_steady_state_tokens_per_second_per_gpu": total / allocation_seconds,
             "measured_rate": None, "feasibility": "unmeasured"}
        ),
        "data_verification": "metadata_only_run_full_staging_before_training",
        "qualification_policy": {
            "minimum_hbm_headroom_fraction": .15, "minimum_paired_speed_improvement_fraction": .03,
            "max_paired_reference_nll_increase_nats_per_token": .02,
            "benchmark_warmup_updates": 5, "benchmark_measured_updates": 20,
            "benchmark_alternating_pairs": 3, "soak_measured_updates": 40,
            "status": "criteria_only_no_passed_evidence",
        },
    }
    return derived, config, report
