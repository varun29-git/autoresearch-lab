# Datasets for thinking preservation + post-CPT SFT (verified 2026-11-13)

## A. Thinking-trace replay during CPT (~5% of mix) + SFT backbone

| Dataset | Rows | License | Why |
|---|---|---|---|
| `nvidia/OpenMathReasoning` (cot split) | 3.2M | CC-BY-4.0 | Largest permissive R1-trace math set; `pass_rate_72b_tir` difficulty signal for curation |
| `nvidia/OpenCodeReasoning-2` | ~2.5M | CC-BY-4.0 | R1 code traces with verification metadata (use `split_0` of v1 for lighter) |
| `open-thoughts/OpenThoughts3-1.2M` | 1.2M | Apache-2.0 | Chat format, `<think>` blocks, domain/difficulty labels — best single drop-in |
| `nvidia/Llama-Nemotron-Post-Training-Dataset` | 1.07M | CC-BY-4.0 | **reasoning on/off system-prompt switch = direct match to our high/low thinking modes** |
| `open-r1/OpenR1-Math-220k` | 225k | Apache-2.0 | Per-generation correctness flags — trivial to filter to verified traces |
| `PrimeIntellect/SYNTHETIC-1-SFT-Data` | 894k | Apache-2.0 | R1 traces with task_type + score |
| `nvidia/AceReason-1.1-SFT` | 213k | CC-BY-4.0 | Math+code R1 traces |

## B. Indic thinking / bilingual traces

| Dataset | Rows | License | Why |
|---|---|---|---|
| `CohereLabs/tiny-aya-l2-thinker-multilingual-reasoning` | ~20.5k Indic (hi/bn/pa/ta/te/ur) | **CC-BY-NC** | Best ready-made Indic thinking traces (question/thinking/output translated + English originals). License flag! |
| `lightblue/reasoning-multilingual-R1-Llama-70B-train` (+`sam2ai` indic/odia subsets) | ~283 Indic + 3,927 Odia | Apache-2.0 / unspecified | Real R1 `<think>` in-language, quality scores per row — style anchor |
| `ai4bharat/indic-gsm8k` | 1,319 × 14 langs | Apache-2.0 | **In-language Indic CoT** (translated answer field). Caveat: it's GSM8K test — don't also eval on it |
| `sarvamai/gsm8k-indic` | 1,319 × 10 langs | unspecified | Indic question + English CoT = natural "English think + Indic answer" seed |

## C. Indic general SFT (fluency, no traces)

| Dataset | Rows | License | Why |
|---|---|---|---|
| `ai4bharat/indic-align` | ~550k+ across splits | CC-BY-4.0 | Canonical Indic alignment corpus, 14 langs, incl. human Anudesh + HHRLHF_T preference |
| `sarvamai/samvaad-hi-v1` | 101k | Apache-2.0 | Natural Hindi conversations |
| `sarvamai/vagartha` | 218k | CC-BY-4.0 | Sanskrit (only if in scope) |

## D. Problem-only sets for self-distillation (run teacher → harvest traces)

- `ai4bharat/MILU-1.5` (CC-BY-4.0, 11 langs, ~73k test + 8k val)
- `sarvamai/mmlu-indic` (MIT, 10 langs)
- `ai4bharat/indic-jee-bench` (MIT), `indic-math-500` (Apache-2.0), `indic-aime-2026` (CC-BY-NC-SA)
- `ai4bharat/indic-gsm8k` / `sarvamai/gsm8k-indic` math problems
- `bharatgenai/BhashaBench-{Ayur,Krishi,Finance,Legal}` domain MCQs; `IndicParam` (CC-BY-NC) low-resource langs

## License watch-outs (commercial launch)
- CC-BY-NC (avoid or renegotiate): tiny-aya thinker, a-m-team sets, KodCode, ServiceNow R1-Distill, facebook/natural_reasoning, IndicParam, indic-aime-2026
- Unspecified/restrictive: pranjalchitale/indicsft (25M rows, no docs), krutrim BhashaKritika (16M, custom license), mlfoundations GeneralThought mirror, sarvamai/gsm8k-indic
- Gated: Nemotron-Post-Training-v2, ai4bharat/MILU (v1), indic-gpqa, FERMAT

## Recommended blend
- CPT replay (5%): OpenThoughts3 + OpenMathReasoning-cot + OpenCodeReasoning-2 + LN-PTD (reasoning on/off) — all CC-BY/Apache
- Post-CPT SFT: LN-PTD + OpenR1-Math-220k (verified-only) + self-distilled bilingual traces (from D) + indic-align + samvaad-hi
- Bilingual thinking style: self-distillation on D problems with the current instruct model (English think → Indic answer), style-anchored on lightblue R1 Indic + ai4bharat/indic-gsm8k CoT
