# Feasibility: 1.5B stack, 1.5T (stretch 2T) tokens in 6,000 H100 GPU-hours

**Question (owner, 2026-10-07):** can the chosen 1.5B architecture train 1.5 trillion tokens (extreme target 2T) within 6,000 H100 GPU-hours, model size fixed? Answer with physics + measured evidence + the lever stack.

## 1. The physics (measured, exact)

Stack 1.5B config (`configs/terminus_25l_2304d_64k_h100_stack.json`, commit `165051a`): 1,500,456,546 params; 9.79 GFLOPs/token (9.00 weights + 0.79 hybrid attention; 65K tied head included in weights). H100 NVL peaks (dense, 400 W): 835.5 TFLOP/s BF16, 1,671 TFLOP/s FP8.

| Rate basis | tok/s/GPU | TFLOP/s | % BF16 peak | % FP8 peak | tokens in 6,000 GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stack, measured today (H100 sweep, mb8×64) | 39,006 | 382 | 45.8% | 22.9% | **0.84T** |
| Reference arch, measured (archived Muon receipt) | 46,395 | 454 | 54.4% | 27.2% | **1.00T** |
| Needed for 1.5T | 69,444 | 680 | 81.4% | 40.7% | 1.50T |
| Needed for 2.0T | 92,593 | 906 | 108.5% | 54.3% | 2.00T |

**Published efficiency ceiling for dense FP8 training at our scale class** (all verified 2026-10-07): FP8-LM GPT-7B 645 TFLOP/s (~39% FP8 peak — the best found, using FP8 gradients + FP8 optimizer states + comms); torchtitan Llama-3.1-8B FP8 410 TFLOP/s; Llama-3-405B BF16 43% BF16-MFU at 8K GPUs; SmolLM3-3B ~26% BF16-equiv; MiniCPM4-1B ~14% BF16-equiv.

**Verdict on raw throughput:** 1.5T-in-6000h (81% BF16-equivalent) exceeds every published dense FP8 run at ≤8B. 2T-in-6000h (108% BF16-equivalent) is beyond demonstrated physics. The repo's own optimistic compute-floor model (~137K tok/s) confirms 69K is *theoretically* inside hardware but requires near-total elimination of non-GEMM time; 93K is past even the floor's credible fraction.

## 2. The lever stack (ranked by size × risk)

### A. Throughput (tokens/s) levers — target ≈53–57K tok/s (from 39K)
1. **Fuse the stack's extra output norms** into the FP8-quant prologue / fused add-norm kernels; recompute in backward (DeepSeek-V3 recomputes all norms) — recovers most of the measured 19% stack-vs-reference gap. [Evidence: Liger 2410.10989; DSv3 2412.19437 App. B]
2. **torch.compile / CUDA graphs on non-GEMM segments** (+15.7% measured at 8B, torchtitan; speedrun full-step capture); the repo already has `terminus/cuda_graph.py`.
3. **FP8 the 65K head GEMM with fused quantization** (+3–6%; speedrun-proven; run a loss-parity ablation — DSv3 keeps the head BF16).
4. **Microbatch/memory retune** (batch 4 gives 45.5K reference at 50 GB — headroom for the stack's buffers), host hygiene (manual GC per OLMo-2 Fig. 16, async prefetch, async checkpointing).
5. **Multi-node:** DDP/ZeRO-1 with bucketed overlapped reduce-scatter preserves ~92–98% per-GPU rate to 64–128 GPUs (torchtitan measured 97–98% at 8B). 6,000 GPU-h on 8 GPUs ≈ 31 days wall-clock.

### B. Token-efficiency levers (quality per token — shrink the target)
1. **Cooldown annealing with a high-quality mixture** (WSD 10% decay; MiniCPM 2404.06395 Table 1 verified: GSM8K +53% rel, MMLU +14% rel at 2.4B; Llama-3 GSM8k +24% rel; OLMo-2 Dolmino corroborates). Largest verified quality-per-token lever. Free.
2. **Mixture search:** RegMix (512×1M proxies ≈ 2–3 H100-h — trivial against 6,000) or DoGE (one 82M×10k proxy); measured 20–50% fewer steps to equal quality (Data-Mixing-Laws 2403.16952: 33%; DoReMi: 2.6× steps but starved math — set a math floor). **Validate against stratified sampling first** (Aioli 2411.05735 negative result).
3. **Dedup** (MinHash + SemDeDup 2303.09540; cross-lingual Indic↔English near-dupes): ~10–20% net tokens.
4. **Educational-quality filtering** (FineWeb-Edu 2406.17557: MMLU +12% rel at equal tokens; FineWeb2 2506.20920 per-language recipe for Indic; Nemotron-CC 2412.02595 warns against over-filtering scarce Indic).
5. **Repetition policy** for scarce Indic: ≤4 epochs ≈ free (2305.16264, verified); code-fill ≤50% harmless.
6. **Architecture dividend already banked:** the stack reaches the baseline's 2.5B loss at ~2.3B tokens (~8% fewer tokens at this horizon) — the architecture work pays into this budget directly.
7. **Tokenizer audit for Indic fertility** (2607.24276: up to 13× fertility tax with wrong tokenizer) — cheap audit; a tokenizer change requires a new campaign version.
8. **Do NOT** do token-level selection (Rho-1) yet: its interaction with FP8/ternary-QAT loss masking is untested.

### C. Budget mechanics
- The dense→ternary QAT stage shares this budget (BitCPM4: QAT ≥40% of total tokens for from-scratch parity; our 500M probe evidence says the stack converts well).
- WSD lets the dense stage pause/resume at full value — schedule insurance if the budget gets re-cut.

## 3. Honest answer

- **1.5T physical tokens in 6,000 GPU-h: no** — not with any published efficiency at this model size (needs 81% BF16-peak-equivalent vs the best published ~65% at 7B with more aggressive FP8 than ours).
- **"1.5T-quality" in 6,000 GPU-h: plausibly yes** — ~1.0–1.2T physical tokens at a fused/compiled ~50–55K tok/s, with the token-efficiency stack (annealing + mixture search + dedup + filtering + the architecture dividend) covering the quality gap. This is the recommended plan.
- **2T physical: beyond demonstrated physics at this size.** 2T-quality: only if the token-efficiency levers over-deliver; treat as stretch, not plan.

## 4. Immediate next steps (no GPU needed except #4)

1. Audit the prepared corpus: Indic fertility under the 65K tokenizer, dedup rate, per-domain token counts. (Controller/CPU.)
2. Run the RegMix/DoGE proxy mixture search on the English-proxy pipeline (hours, not days).
3. Specify the annealing-phase mixture (high-quality pool + task-adjacent data) for the WSD decay.
4. Stage three on the H100: fuse the output norms + compile the pointwise segments + FP8 head ablation; re-measure against the 46.4K reference and the 69.4K requirement.
5. Pilot-validate the data levers at ~100–300B tokens before committing the full run (the Aioli lesson).

Sources: full-text-verified list in the campaign research log (`notes/research-log-20261006.md`) plus the 2026-10-07 MFU and token-efficiency searches (per-source verification status on file in session records; headline IDs inline above).
