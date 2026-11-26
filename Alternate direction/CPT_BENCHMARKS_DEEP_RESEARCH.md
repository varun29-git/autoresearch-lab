# Deep research: typical sizes & mixture proportions of language-adaptation CPT runs
Compiled 2026-11-12. All numbers verified against primary sources (arXiv, HF model cards, official blogs) by four parallel research agents. Purpose: calibrate the Spark-X2.6-1.7B → Indic CPT budget (candidate: 60–200B tokens, 75–80% Indic / 20–25% replay).

## 1. The dataset: ~45 projects, 2023–2026

### 1a. East & Southeast Asia
| Project | Base | CPT tokens | Mixture (target/replay) | Vocab changed | Outcome |
|---|---|---|---|---|---|
| Swallow (arXiv 2404.17790) | Llama-2 7/13/70B | ~100B | 90% JA / 10% EN | Yes (+11k) | JA +7 avg; EN −2–5; unsaturated at 100B |
| Sailor (2404.03608) | Qwen1.5 0.5–14B | 200B (400B for 0.5B) | 70% SEA / 30% EN+ZH | No | raising SEA share always hurt EN; forgetting controlled by replay+LR |
| Sailor2 (2502.12982) | Qwen2.5 1/8/20B | 500B | 80% SEA / 20% replay | No | 20B ≈ GPT-4o 50-50 on SEA |
| SEA-LION v3 (HF) | Llama-3.1-70B-Inst | 200B | ~42% SEA / 58% EN+ZH+code | No | top open SEA model |
| SEA-LION v4 (HF) | Gemma-3-27B-IT | ~500B | 41% SEA / 59% replay | No | best open <200B SEA |
| EEVE-Korean (2402.14714) | SOLAR-10.7B | ~1.6B | Korean-dominant | Yes (+9k) | KoBEST BoolQ .51→.86, EN preserved/improved |
| Typhoon v1 (2312.13951) | Mistral-7B | undisclosed | 50/50 | Yes (+5k) | ThaiExam ≈ GPT-3.5 |
| Typhoon 2 (2412.13702) | Llama-3 1–70B | undisclosed (pool 44B) | 50/50 | **No** (degradation risk) | base choice > mixture |
| Komodo (2403.09362) | Llama-2-7B | 8.8B ×3 ep (LoRA) | ID+regional+EN | Yes (+3k) | EN ppl ≈ equal |
| Chinese-LLaMA (2304.08177) | LLaMA-1/2 | 20–120 GB text | ~100% ZH (LoRA) | Yes (+20k) | moderate gains |
| Nemotron-Nano-9B-v2-Japanese | Nemotron-Nano-9B | undisclosed | JA+EN | No | Nejumi LB4 .711 vs Qwen3-8B .690 |

### 1b. Europe / Middle East / multilingual
| Project | Base | CPT tokens | Mixture | Vocab changed | Outcome |
|---|---|---|---|---|---|
| Occiglot-7B-EU5 | Mistral-7B | 293B | ~80% EU4 / 13% EN / 5% code | No | EU gains, slight EN regression |
| TowerLLM (2402.17733) | Llama-2 7/13B | 20B | 2/3 mono + 1/3 parallel | No | 85% of gains by 5B tokens |
| Latxa (2403.20266) | Llama-2 7–70B | 10B | 82% Basque / 18% EN | No | 70B +25 pts Basque; replay ablation +13 Belebele |
| Bielik-11B-v2 (2505.02410) | Mistral-7B→11B | 198B | PL-heavy + EN replay | No | PL leaderboard 65.7; EN +4 |
| Krikri (2505.13772) | Llama-3.1-8B | 110B | 62% GR / 23% EN / 6% parallel / 9% math+code | Yes (GR ext.) | +10.8% GR, +0.8% EN |
| DictaLM 2.0 (2407.07080) | Mistral-7B | ~200B | ~50% HE / ~50% EN | Yes (+1k, emb. distill) | SOTA Hebrew |
| AceGPT (2309.12053) | Llama-2 7/13B | 30B / 10B | 64/36 and 60/40 | No (cost) | best open Arabic MMLU at release |
| ALLaM (2407.15390) | Llama-2 7–70B | 1.2T (600B 70B) | 45% AR / 55% EN (ablated 6 ratios) | Yes (merge, sub-token-avg init) | no catastrophic EN forgetting |
| Fanar-1-9B (2501.13944) | Gemma-2-9B | 1T | 40% AR / 50% EN / 10% code | No | MMLU/ArabicMMLU both up |
| openCabrita (2308.11878) | OpenLLaMA-3B | 7B | 100% PT, 0 replay | Yes (merge) | PT tasks up, "slight" EN decrease |
| (from-scratch refs) | EuroLLM-1.7B 4T (50/45/5), Salamandra 2–40B 9–12.9T, Teuken-7B 4T (42% EN), CroissantLLM-1.3B 3T (1:1), Jais-13B 395B (~33% AR) |

### 1c. Indic
| Project | Base | CPT tokens | Mixture | Vocab changed | Outcome |
|---|---|---|---|---|---|
| OpenHathi-7B (Sarvam) | Llama-2-7B | ~3B phase 1 (LoRA) | ~50/50 bilingual | Yes (+16k) | small EN degradation |
| Tamil-LLaMA (2311.05845) | Llama-2 7/13B | 12 GB text, LoRA | ~100% TA | Yes (+16k) | IndicSentiment .51→.81 |
| Kan-LLaMA / Ambari | Llama-2-7B | ~0.5–0.6B, LoRA | ~100% KN | Yes/No | demos only |
| **Llama-3-Nanda-10B** (2504.06011) | Llama-3-8B +expansion | ~110B | **1:1 HI:EN — "relatively high replay necessary"** | Yes (weighted-avg init) | SOTA Hindi 10B |
| **Nemotron-Mini-Hindi-4B** (2410.14815) | Nemotron-Mini-4B | **400B** | **50% HI / 50% EN replay** | No (256k vocab) | SOTA Hindi 4B; EN "competitive" |
| **NVIDIA NPI-v1** (2610.26006) | Nano/Lightning-3.5 | **100B @45% Indic; 200B @10/20/30%** | remainder Nemotron mix | No | Indic mean 63.1→65.6 monotonic; EN flat/mixed |
| **Nemotron 3.5 Super VL** (shipped) | Super 120B-A12B | 100B stage | 10% Indic / 90% replay | No | MILU +5.4, ARC-C +8.7, GSM8K +9.8, no EN regression |
| **Gnani.AI** (2610.15373) | Nemotron-3-Nano-30B-A3B | **1.27T** (+3B emb warmup) | 90% Indic / 10% EN | **Yes: 131,072→163,072 continued-BPE** | MILU +12.9, ARC-Indic +13.0; **MMLU −3.4, MMLU-Pro −9.4** |
| (from-scratch refs) | Sarvam-1 2B/4T (50% Indic); Krutrim-1 7B/2T (recommends CPT 75/25); BharatGen Param-1 2.9B/5T+2T (phase-2 50% Indic + 20% replay); Sarvam-30B/105B 16T/12T |

### 1d. Academic / theory anchors
- **Ibrahim et al. 2024** (2403.08763): LR rewarm+redecay + replay **matches retraining-on-union**; 5% replay enough for weak shift, **25% for strong language shift**; sweep 0/10/25/50%.
- **D-CPT Law** (2406.01375): fitted optimal general-data share **8–27%** across domains.
- **Gupta et al. 2023** (2308.04014): CPT with rewarm works up to 297B (100% of upstream); max-LR 0.5–2× original trades adaptation vs forgetting.
- **Dagan et al. 2024** (2402.01035): tokenizer swap becomes performance-neutral **>50B tokens**; embedding-only training 100–500B hurts.
- **ZeTT** (2405.07883): with learned init, **<1B tokens** closes the tokenizer-swap gap.
- **Ravichandran et al. 2024** (2402.17400): small models learn *and forget* fastest; small-budget CPT can hurt well-pretrained models.
- **Stability gap** (2406.14833): expect initial dip then recovery; 40% of naive budget suffices with HQ-data × epochs; small models need higher LR (1.1B → 3e-4).
- **AfriqueLLM** (2601.06395): 26B @ ~85% target / 15% HRL replay + code/math; monolingual-only CPT **degrades reasoning**; mixture is the dominant factor.
- **DeepSeek-V2** used 30% replay on 6T CPT; **Zamba-2** 60% replay in 50B anneal (via Ibrahim §2).

## 2. Distributions — what "typical" actually is

**Token budget (serious full-parameter language CPT):**
- Hobbyist LoRA era (2023): 0.1–3B tokens
- **Standard serious CPT: 55–500B tokens** (Swallow 100B, Krikri 110B, Nanda 110B, Bielik 198B, DictaLM 200B, Sailor 200B, SEA-LION 200B, Occiglot 293B, Nemotron-Mini-Hindi 400B, Sailor2/SEA-LION-v4 500B)
- Lab/national scale: 0.6–1.3T (Fanar 1T, ALLaM 1.2T, Gnani 1.27T)
- As % of upstream pretraining: typically 5–25%; academic domain-CPT works at 0.1–10%.

**Target-language share:**
- Forgetting-conscious bilingual: **45–65%** (ALLaM 45, Fanar 40, DictaLM ~50, AceGPT 60–64, Krikri 62, Nanda 50, Nemotron-Mini-Hindi 50, BharatGen 50, NPI-Nano 45)
- Deep adaptation: **70–90%** (Sailor 70, Sailor2 80, Occiglot ~80, Latxa 82, Swallow 90, Gnani 90, AfriqueLLM 85)
- Production big-model ships: 10–50% (Super VL 10, NVIDIA 10–30, SEA-LION ~41)

**Replay share:** evidence-backed **5–25%** (Ibrahim: 5% weak shift / 25% strong shift; D-CPT optima 8–27%; AfriqueLLM 15%); Indic-specific practice clusters higher (**50%**: Nanda, Nemotron-Mini-Hindi; Krutrim floor 25%); **no successful Indic project used <10% replay**.

**Tokenizer surgery:** needs >50B CPT to be neutral with naive init, <1B with learned init; 2023 projects always did surgery, 2024+ big-vocab projects often skip it. Our Spark case must do it (3.9% Indic vocab) — Gnani is the direct precedent (continued-BPE 131k→163k, row-freezing, AdaptMix).

**Retention cost without care:** −3.4 MMLU / −9.4 MMLU-Pro at 90% target (Gnani, 1.27T); −10% HRL at 85% target no-replay-ish (AfriqueLLM Llama); −10.2% HRL. With 20–50% replay: flat to +1 English (Krikri +0.8, Fanar +0.7, EEVE +, NVIDIA flat).

## 3. Where the Spark-Indic plan sits

Candidate plan: **200B tokens, 75% Indic / 25% EN+code replay, 1.7B dense**.

- Budget 200B = dead-center of the standard serious-CPT cluster (55–500B); ~matches Swallow/Sailor/DictaLM/SEA-LION/Occiglot/Bielik; well below Gnani's 1.27T overkill.
- Target share 75–80% = aggressive but precedented (Swallow 90, Latxa 82, Occiglot 80, Sailor2 80, Gnani 90). Justified here because the base has the weakest Indic substrate of any case studied (3.9% vocab, MILU 30–35 vs 25 random).
- Replay 20–25% = exactly the Ibrahim/D-CPT evidence-backed band for a **strong** distribution shift. Because 1.7B is small and small models forget fastest (Ravichandran), hold replay at the **top** of the band (25%), include code+math in replay (AfriqueLLM showed reasoning degrades without it), and anneal the final 10–20B.
- LR: rewarm + redecay at 0.5–1× original peak; small model → don't under-LR (stability-gap paper: 1.1B needed 3e-4).
- Expected outcome by analogy: +10–25 points Indic benchmarks, English within ±2 — matching Swallow/Krikri/Fanar/NVIDIA patterns.

## 4. Caveats
- Many projects (Typhoon, SeaLLM v1/2, Vistral, Sabiá, Nemotron-JP, Krutrim) don't disclose token counts; the disclosed set may skew toward larger budgets.
- Fertility differences make "tokens" non-comparable across tokenizers (NVIDIA flags this: multiMixV8 vs Gemma-4 counts). Our plan counts in post-surgery Spark tokens.
