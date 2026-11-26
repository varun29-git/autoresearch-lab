# Fixed Stack 4 profile for Kimi's optimizer research

This is a factual description of the model and experiment contract. Kimi must
research the literature and released implementations independently, then
choose and test learning rates, weight decay, and optimizer-group settings.
Numbers already present in `reference_200m.json` keep the training config
parseable; they are **unselected placeholders**, not recommended rates. The
candidate must explicitly provide all six core LR/decay values, so those
reference rates cannot silently become the first trial's choices.

| Property | Current 200M dense HPO model |
| --- | --- |
| Parameters | 200,290,866 |
| Architecture | Stack 4; 25 transformer layers; Peri-LN residual branches and learned value residual |
| Hidden / FFN width | 768 / 1,920 |
| Attention | 6 query heads, 2 KV heads, head dimension 128; local 1,024-token attention with every fifth layer global |
| Context / tokenizer | 4,096 tokens / fixed 65,536-token vocabulary |
| Embedding | Input embedding and output classifier share one weight table |
| Dense training | FP8 DeepGEMM linears, BF16 residuals, full FP8 backward; no ternary QAT |
| Data / comparison | Fixed prepared English stream and fixed held-out English cross-entropy |
| Batch / schedule | 393,216 tokens per optimizer update; archived 2.5B-token WSD run warms up over ~250M tokens, then stays at peak LR until decay begins near 2B tokens |
| Trial endpoint | 1,272 updates = 500,170,752 tokens per complete candidate |

The optimizer implementation currently assigns these parameter roles:

| Role | Current update algorithm | Independently tunable fields |
| --- | --- | --- |
| Packed logical Q/K/V projection | Muon, logical-slice Newton–Schulz | `muon_lr`, optionally `muon_qkv_lr` |
| Attention output projection | Muon | `muon_lr`, optionally `muon_o_lr` |
| Packed gate/up projection | Muon, logical-slice Newton–Schulz | `muon_lr`, optionally `muon_gate_up_lr` |
| FFN down projection | Muon | `muon_lr`, optionally `muon_down_lr` |
| Tied embedding/classifier | Auxiliary AdamW | `adamw_lr`, `embedding_weight_decay` |
| Input norms; Peri-LN gains; final norm | Auxiliary AdamW | `norm_lr`, optionally `peri_norm_lr` and `final_norm_lr` |
| Q/K normalization gains | Auxiliary AdamW | `qk_norm_lr` |
| Learned value-residual mixing | Auxiliary AdamW | `value_mix_lr` (otherwise `norm_lr`) |

`weight_decay` controls Muon matrices and any other AdamW matrix not given its
own override. **Optimizer weight decay acts from the first update**: a group
with learning rate `η_t` and weight decay `λ` multiplies its weights by
`1 − η_t λ` each step. This is different from the WSD learning-rate decay
scheduled near 2B tokens. The tied table can use a separate
`embedding_weight_decay`; Kimi must choose and test both values.

The selected Muon rule is fixed for this campaign: `spectral_unclamped`.
A logical matrix receives its Newton–Schulz direction multiplied by
`√(d_out/d_in)` and its chosen LR. The rule is a width-transfer hypothesis,
not a complete μP guarantee for the tied embedding and readout.
The runner fixes the model, data, tokenizer, batch, schedule, and scoring;
candidate changes are optimizer hyperparameters and global clipping only.
