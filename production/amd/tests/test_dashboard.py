import json
import time

from production.amd.dashboard import summarize


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_dashboard_reads_trainer_files_and_flags_a_stale_heartbeat(tmp_path):
    now = 1_800_000_000
    run = tmp_path / "run"
    run.mkdir()
    _write(run / "events.jsonl", [
        {"time_unix": now - 100, "event": "startup", "counts": {"trainable": 10}, "world_size": 8,
         "state": {"updates": 0, "tokens_seen": 0}},
        {"time_unix": now - 40, "event": "train", "updates": 12, "tokens_seen": 25_165_824,
         "loss": 8.5, "grad_norm_before_clip": 1.2, "grad_clip_coefficient": 1.0,
         "lr_scale": 0.02, "tokens_per_second": 1000, "data_wait_seconds": 0.1,
         "peak_memory_bytes": 2 * 1024**3, "quantization_strength": 0, "valid_targets": 2_097_152,
         "optimizer_group_lr_min": 1e-4, "optimizer_group_lr_max": 0.001},
        {"time_unix": now - 20, "event": "validation", "updates": 12, "tokens_seen": 25_165_824,
         "languages": {"en": {"loss": 8.0, "tokens": 100, "perplexity": 2980.0},
                       "hi": {"loss": 8.4, "tokens": 100, "perplexity": 4447.0}},
         "mixture": {"loss": 8.1, "tokens": 200, "perplexity": 3294.0},
         "benchmarks": {"mmlu": {"correct": 15, "items": 100, "accuracy": 0.15, "mean_margin": 0.02}}},
    ])
    _write(run / "incidents.jsonl", [
        {"time_unix": now - 80, "event": "token_count_mismatch", "valid_targets": 3, "update_tokens": 4},
    ])
    _write(run / "stability.jsonl", [
        {"time_unix": now - 30, "event": "stability", "updates": 96, "probes": {
            "layer_00.residual": {"rms": 0.4, "nonfinite_fraction": 0.0},
            "layer_00.attention_sample": {"mean_entropy": 2.5, "max_probability": 0.2},
            "layers.0.attn.q_proj.fp8_input": {"relative_rms_error": 0.01, "nonfinite_fraction": 0.0},
            "embed.parameter_sample": {"update_to_weight_rms": 0.003},
        }},
    ])
    (run / "heartbeat.json").write_text(json.dumps({
        "schema": 1, "time_unix": now - 1000, "status": "running",
        "budget": {"remaining_gpu_hours": 5000, "elapsed_gpu_hours": 10, "training_seconds": 100},
    }))
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "latest.json").write_text(json.dumps({
        "file": "step_00000012.pt", "size_bytes": 1000, "state": {"updates": 12, "tokens_seen": 25_165_824},
    }))
    report = summarize(run, now=now)
    assert report["status"] == "stale"
    assert report["latest_train"]["loss"] == 8.5
    assert report["series"]["loss"][-1]["y"] == 8.5
    assert report["validations"][-1]["benchmarks"]["mmlu"]["accuracy"] == 0.15
    assert report["stability"]["summary"]["max_fp8_relative_rms"] == 0.01
    assert report["stability"]["layers"][0]["entropy"] == 2.5
    assert report["incidents"][0]["event"] == "token_count_mismatch"
    assert report["operations"]["checkpoint_updates"] == 12
    assert report["operations"]["remaining_gpu_hours"] == 5000
    assert report["progress"]["band"].startswith("Warmup")
    assert any("900" in banner for banner in report["banners"])
    by_task = {row["task"]: row for row in report["benchmarks"]}
    assert by_task["mmlu"]["state"] == "scored"
    assert by_task["mmlu"]["accuracy"] == 0.15
    for task in ("arc-challenge", "arc-easy", "belebele", "boolq", "global-mmlu",
                 "hellaswag", "include", "piqa", "winogrande", "gsm8k", "humaneval",
                 "indic-humaneval", "indic-mbpp", "indicgenbench-flores",
                 "indicgenbench-xorqa", "indicqa", "math", "mbpp", "mgsm",
                 "scalingbench/bbh", "scalingbench/human_eval", "scalingbench/math",
                 "scalingbench/mbpp", "scalingbench/mmlu"):
        assert by_task[task]["state"] == "waiting"
    assert by_task["milu"]["state"] == "not_tokenized"
    assert by_task["facebook/flores"]["state"] == "not_tokenized"
    empty = summarize(tmp_path / "absent", now=now)
    assert empty["status"] == "waiting"
    assert empty["banners"] == []
    assert all(row["state"] != "scored" for row in empty["benchmarks"])
