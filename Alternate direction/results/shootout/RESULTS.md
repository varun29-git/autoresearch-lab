# Spark-X2.6-1.7B vs MiniCPM5 — independent benchmark shootout
Measured 2026-11-13, three separate A10G VMs, identical codepath/settings for all models:
temp 1.0, top_p 0.95, do_sample, thinking=high (chat template), single script
(`scripts/fast_bench.py`, `scripts/gpqa_only.py`). Extraction: boxed/last-int (GSM8K, AIME),
boxed- letter (GPQA-D via hendrydong/gpqa_diamond_mc mirror, `test` split).

## Headline table (measured vs vendor-claimed)

| Benchmark | Spark measured | Spark claimed | MiniCPM5-1B measured | MiniCPM5-2B measured | MCP5-2B claimed |
|---|---|---|---|---|---|
| GSM8K | 54.8 | — | 48.7 | **73.8** | — |
| AIME 2025 (30×4) | 1.7 | — | 5.0 | **19.2** | — |
| AIME 2026 (30×4) | 0.83 | **70.1** | — | — | 86.5 |
| GPQA-Diamond | 14.1 | **56.1** | 7.6 | **95.96** | 70.2 |
| A10G decode b1 tok/s | 23.3 | — | — | 25.9 | — |
| A10G decode b16 aggregate | 361 | — | — | 405 | — |
| A10G prefill tok/s | 7,589 | — | — | 8,535 | — |

Caps: GSM8K 1024, GPQA 2048, AIME 4096 new tokens (identical all models).
Spark-only earlier runs at 8192 cap: AIME25 6.7%, AIME26 0.83%.

## Verdicts on Spark's vendor claims
- **AIME-2026 70.1: NOT reproduced.** 0.83% measured (30 problems × 4 samples, official
  math-ai/aime26). Off by ~85×. AIME25 also weak (1.7–6.7%).
- **GPQA-Diamond 56.1: NOT reproduced.** 14.1% measured — below the 25% random floor.
  (Partly truncation/extraction: verbose thinking hits the 2048 cap and no letter is emitted;
  same method applied to all models.)
- **GPQA-Diamond MiniCPM5-2B 95.96%: suspiciously HIGH** — above frontier-model territory;
  GPQA-like data is very likely inside MiniCPM5's UltraData training mix. Treat their 70.2
  card number the same way — contaminated either way.
- **GSM8K**: MiniCPM5-2B 73.8 >> Spark 54.8 > MiniCPM5-1B 48.7.
- **Agentic claims (SWE-bench 59.8, BFCL 55.2, τ² 85.3, MCP-Atlas 41.8)**: not runnable in
  this harness class (multi-day agentic setups: Docker-per-instance, simulated users, MCP
  server farms). Unverified; on the evidence of the verifiable rows, treat as marketing.
- **Throughput**: MiniCPM5-2B modestly faster per GPU (smaller head); both slow under plain
  HF backend (no vLLM for these custom archs out of the box).

## Bottom line for the Indic project
On general reasoning, MiniCPM5-2B beats Spark-X2.6-1.7B decisively under identical
conditions. Spark's remaining value is surgical (26% cleanly-prunable Chinese vocab,
tied embeddings, byte-level BPE with CJK isolation, Apache-2.0, 3.44GB footprint).
Base-model choice should be decided on tokenizer composition + Indic substrate, not
vendor benchmarks: compare MiniCPM5's vocab before committing.
