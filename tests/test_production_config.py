import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from production.build_dense_config import build_config
from terminus.data import TokenDataConfig, TokenManifest
from terminus.preflight import count_model
from terminus.model import BitNetLlama
from terminus.train import load_config


def test_recipe_changes_only_reviewed_architecture_fields(tmp_path):
    # Historical reference is test evidence only, never a production dependency.
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "hpo/reference_200m.json").read_text())["config"]
    old["model"].update(dim=2304, ffn_dim=5760, n_heads=18, n_kv_heads=6)
    old["optimizer"] = dict(kind="muon", muon_lr_rule="spectral_unclamped",
                            muon_fan_in_base=None, muon_lr=.017, adamw_lr=.003,
                            norm_lr=.0003, qk_norm_lr=.0001, weight_decay=.02,
                            embedding_weight_decay=0.)
    old["training"].update(gradient_accumulation_steps=8, warmup_tokens=20_000_000,
                           decay_tokens=100_000_000, checkpoint_interval=1000,
                           validation_interval=1000, validation_batches_per_language=8,
                           log_interval=10, stability_log_interval=10, time_limit_seconds=None)
    new = build_config(manifest=manifest(), world_size=32, global_batch_tokens=4_194_304,
                       warmup_tokens=20_000_000, decay_tokens=100_000_000)
    assert {key: value for key, value in new["model"].items()
            if value != old["model"].get(key)} == {
        "n_layers": 24, "global_attention_every": 6, "local_attention_window": 512}
    old["model"].update(n_layers=24, global_attention_every=6, local_attention_window=512)
    normalized = []
    for name, config in (("old", old), ("new", new)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(config))
        normalized.append([asdict(section) for section in load_config(path)])
    assert normalized[0] == normalized[1]


def manifest(*, batch_size=4, seq_len=4096, total_tokens=1_000_000_000):
    return TokenManifest(TokenDataConfig(batch_size=batch_size, seq_len=seq_len,
                                         total_train_tokens=total_tokens), [])


def test_selected_dense_recipe_and_global_batch(tmp_path):
    config = build_config(manifest=manifest(), world_size=32,
                          global_batch_tokens=4_194_304,
                          warmup_tokens=20_000_000, decay_tokens=100_000_000)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    model, training, optimizer = load_config(path)
    assert count_model(model)["total"] == 1_446_478_126
    assert (model.n_layers, model.dim, model.ffn_dim, model.n_heads, model.n_kv_heads) == (24, 2304, 5760, 18, 6)
    assert (model.global_attention_every, model.local_attention_window, model.max_seq_len) == (6, 512, 4096)
    with torch.device("meta"):
        instance = BitNetLlama(model)
    assert [index + 1 for index, layer in enumerate(instance.layers)
            if layer.attn.is_global_layer] == [6, 12, 18, 24]
    assert (model.branch_norm, model.value_residual, model.weight_mode) == ("peri", "learned", "dense")
    assert (optimizer.muon_lr_rule, optimizer.muon_lr, optimizer.adamw_lr) == ("spectral_unclamped", 0.017, 0.003)
    assert (optimizer.weight_decay, optimizer.embedding_weight_decay) == (0.02, 0.0)
    assert training.gradient_accumulation_steps == 8
    assert training.warmup_tokens == 20_000_000 and training.decay_tokens == 100_000_000


def test_bad_batch_or_schedule_is_rejected():
    kwargs = dict(manifest=manifest(), world_size=32, global_batch_tokens=4_194_304,
                  warmup_tokens=20_000_000, decay_tokens=100_000_000)
    with pytest.raises(ValueError, match="multiple"):
        build_config(**(kwargs | {"global_batch_tokens": 4_194_305}))
    with pytest.raises(ValueError, match="fit"):
        build_config(**(kwargs | {"decay_tokens": 1_000_000_000}))
    with pytest.raises(ValueError, match="4096"):
        build_config(**(kwargs | {"manifest": manifest(seq_len=2048)}))
