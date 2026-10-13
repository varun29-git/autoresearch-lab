"""Selected dense baseline, independent of archived research campaigns.

These are model/optimizer choices, not a runnable production configuration.
Run-dependent data, batch and schedule settings belong outside the repository.
"""

SELECTED_MODEL = {
    "attention_backend": "fa3",
    "branch_norm": "peri",
    "dim": 2304,
    "ffn_dim": 5760,
    "fp8_backward": "full",
    "fp8_cache_weights": True,
    "fp8_quantizer": "triton_split",
    "fp8_wgrad_splits": 0,
    "fuse_projections": True,
    "fused_mlp": True,
    "fused_pointwise": True,
    "fused_qkv": True,
    "global_attention_every": 5,
    "linear_backend": "deepgemm",
    "local_attention_window": 1024,
    "loss_backend": "triton",
    "loss_chunk_size": 4096,
    "max_seq_len": 4096,
    "n_heads": 18,
    "n_kv_heads": 6,
    "n_layers": 25,
    "pad_token_id": 3,
    "qk_norm_mode": "learned_rms",
    "quantize_activations": True,
    "residual_dtype": "bf16",
    "residual_init_scale": 0.14078571632817444,
    "use_subln": False,
    "value_residual": "learned",
    "vocab_size": 65536,
    "weight_init": "normal",
    "weight_mode": "dense",
    "weight_scale_layout": "blockwise",
}

SELECTED_OPTIMIZER = {
    "kind": "muon",
    "muon_lr_rule": "spectral_unclamped",
    "muon_fan_in_base": None,
    "muon_lr": 0.017,
    "adamw_lr": 0.003,
    "norm_lr": 0.0003,
    "qk_norm_lr": 0.0001,
    "weight_decay": 0.02,
    "embedding_weight_decay": 0.0,
}

SELECTED_TRAINING = {
    "compile": False,
    "decay_style": "cosine",
    "development_mode": False,
    "device": "cuda",
    "dtype": "bf16",
    "final_lr_fraction": 0.1,
    "grad_clip": 1.0,
    "schedule": "wsd",
    "seed": 1337,
}
