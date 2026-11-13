# Conservative CPT Recipes for Spark's Indic Adaptation

## Executive Summary

- **Keep Spark Intact**: Preserve the supplied tokenizer, tied embeddings, gated attention, hybrid masks and RoPE configuration. Nanda and Sailor2 gain capacity through block expansion, so their success is not evidence for modifying this fixed Spark checkpoint. [7][46] -> Use ordinary full-parameter CPT, not architecture surgery.
- **Screen Learning Rates, Not Assumptions**: Published adaptation rates span DictaLM 2's **5e-6**, Sailor's **1e-4** and Hindi Mini's **2e-4**. [22][19][31] -> Screen **1e-5, 3e-5 and 1e-4** as an explicitly extrapolated Spark grid; do not call any of them Spark's original LR.
- **Use A Conditional Relative Cap**: Ibrahim tests half, equal and double the original pretraining peak; lower peaks favor retention. [43] -> My conservative starting target is **0.5 times the original peak**, an engineering choice rather than a universal law. Spark's unpublished original LR prevents numerical calibration of that fraction.
- **Keep The Fixed Replay Share**: Hindi Mini uses **200B Hindi + 200B English**, and Typhoon 2 uses **50% English** during CPT. [31][45] -> The requested 50/50 mixture has direct precedent, but it does not guarantee English retention.
- **Rank Fully Decayed Pilots**: Ibrahim fits cosine decay to the token budget; Gupta's higher peaks improve adaptation while increasing forgetting. [43][16] -> Give every LR arm the same complete pilot horizon, then restart the selected long-run recipe from the released base.
- **Prefer Simple Optimizer Controls**: OLMo 2 excludes embeddings from weight decay after observing unhealthy embedding-norm behavior with decay. [42] -> Use AdamW with a separate zero-decay tied-embedding group; treat its transfer to Spark as an extrapolation, not a matched CPT result.
- **Expect A Real Retention Trade-Off**: Hindi Mini's base-to-base English MMLU falls **2.23 percentage points** and ARC-Challenge falls **4.82 points**, calculated from its reported scores. [31] -> Set acceptance thresholds before training; these observed losses are not a recommended tolerance.
- **Separate Official SFT From CPT**: XHToken's fork publishes Spark-specific full-SFT examples at **4e-5**, with **Muon enabled**, but these are not pretraining recipes. [51][53] -> Use the fork for model compatibility, not as proof that its optimizer or zero warmup is appropriate for CPT.


## 1. Indic Evidence: Useful Recipes, Important Mismatches

**Scope and notation.** This report uses primary papers, model cards, official articles and released code available by **November 12, 2026**. The model specification in the question is accepted as fixed. **NR** means not reported in the located primary source, not zero and not a framework default. **Derived** means arithmetic from published values. A corpus size is not automatically a consumed-token budget. Parameter-update scope is marked NR when the source does not explicitly distinguish full training from parameter-efficient training.

### Nanda's replay result transfers better than its architecture

Nanda starts with Llama-3-8B, inserts blocks to reach a 10B model, and freezes the original backbone while training the added blocks. Its reported best Hindi adaptation mix is **1:1 English:Hindi**. This supports the requested replay share, but not the claim that its LR is validated for full-parameter adaptation of an unchanged 1.7B model. [7]

Hindi Mini is the more relevant unchanged-tokenizer Indic case: its CPT consumes **400B tokens**, equally Hindi and English, with cosine decay from **2e-4 to 4.5e-7**. Nevertheless, English scores deteriorate on several tasks. Large replay and a large initial vocabulary therefore reduce one adaptation difficulty without eliminating the stability-plasticity trade-off. [31]

### Indic run ledger: identity, consumption and parameter scope

| Run | Base size / resulting size | CPT tokens consumed | Target / replay mixture | Full vs LoRA | Vocabulary changed? |
|---|---|---|---|---|---|
| Llama-3-Nanda | Llama-3 8B -> 10B | 55B Hindi in the adaptation comparison; final total NR; 110B is only a conditional derivation from 1:1 | 50% Hindi / 50% English | Added blocks trained; original backbone frozen | Yes, 128256 -> 153856 [7] |
| Nemotron-Mini-Hindi-4B | Nemotron Mini; architecture table 4.19B; text also describes 2.6B trainable parameters | 400B | 200B Hindi / 200B English | CPT; exact update scope NR | No; existing 256k tokenizer [31] |
| NPI Nano data comparison | Nemotron Nano; exact parameter size NR | 100B | 45B Indic / 55B non-Indic | NR | NR [9] |
| NPI Nano share study: 10% | Nemotron Nano; exact size NR | 200B | 20B Indic / 180B other, derived | NR | NR [9] |
| NPI Nano share study: 20% | Nemotron Nano; exact size NR | 200B | 40B Indic / 160B other, derived | NR | NR [9] |
| NPI Nano share study: 30% | Nemotron Nano; exact size NR | 200B | 60B Indic / 140B other, derived | NR | NR [9] |
| NPI Super VL example | Nemotron 3.5 Super VL; outside the requested small-dense comparison | 100B | 10B Indic / 90B preceding training mixture | NR | NR [35] |
| Gnani Evon v3.3 | Nemotron 3 Nano-derived 30B MoE, 3.3B activated; not dense 3B | About 1.27T: 974B broad + 195B annealing + 100B long-context | Final mixture NR; separate 120B study uses 45% Indic / 55% English | CPT update scope NR; earlier embedding-only stage freezes backbone | Yes, 131072 -> 163072 [8] |

**Takeaway:** Hindi Mini supplies the cleanest named Indic precedent for an unchanged vocabulary and equal replay. Nanda and Gnani are valuable selection and failure cases, but their structural interventions make direct LR transfer hazardous.

### Indic schedules and batches

| Run | Published LR | Scheduler | Warmup | Final LR | Global batch, tokens/update | Sequence length |
|---|---|---|---|---|---|---|
| Nanda | Peak 1.5e-4 | Linear warmup, cosine decay | 1% of steps | 1.5e-5, derived from 10x decay | 4M | 8192 [7] |
| Hindi Mini | Start 2e-4 | Cosine | NR | 4.5e-7 | NR | NR [31] |
| NPI Nano, 100B | NR | Learning-rate decay phase; shape NR | NR | NR | NR | NR [9] |
| NPI Nano, 200B, 10% | NR | Decay phase; shape NR | NR | NR | NR | NR [9] |
| NPI Nano, 200B, 20% | NR | Decay phase; shape NR | NR | NR | NR | NR [9] |
| NPI Nano, 200B, 30% | NR | Decay phase; shape NR | NR | NR | NR | NR [9] |
| NPI Super VL | NR | Decay-phase CPT; exact shape NR | NR | NR | NR | NR [35] |
| Gnani Evon | Selected peak 1e-5; rejected aggressive candidate 1e-4 | NR | NR | NR | NR | Final phase 131072; earlier lengths NR [8] |

NPI is evidence about consumed budgets and Indic allocation, not a published optimizer recipe. Its 2026 paper and official blog describe different experiments; the Nano 45% Indic comparison must not be combined with the Super VL 10% Indic example into a fictitious common run. [9][35]

### Indic optimizer and embedding details

| Run | Optimizer | Adam betas | Weight decay | Gradient clip | Embedding LR | Embedding freeze during CPT |
|---|---|---|---|---|---|---|
| Nanda | AdamW, epsilon 1e-5 | 0.9, 0.95 | 0.1 | 1.0 | NR | NR separately; original backbone frozen [7] |
| Hindi Mini | Refers to Parmar et al. optimizer settings; numeric settings not restated | NR in this paper | NR | NR | NR | NR [31] |
| NPI Nano, 100B | NR | NR | NR | NR | NR | NR [9] |
| NPI Nano, 200B, 10% | NR | NR | NR | NR | NR | NR [9] |
| NPI Nano, 200B, 20% | NR | NR | NR | NR | NR | NR [9] |
| NPI Nano, 200B, 30% | NR | NR | NR | NR | NR | NR [9] |
| NPI Super VL | NR | NR | NR | NR | NR | NR [35] |
| Gnani Evon | AdamW is discussed explicitly | NR | Numeric value NR | NR | NR | CPT policy NR; inherited rows frozen in the earlier new-token stage [8] |

The missing values should remain missing. In particular, Hindi Mini's SFT and DPO batch sizes are not CPT batch sizes. [31]

### Gnani, Sarvam and Krutrim: three different lessons

Gnani's sweep initially favored **1e-4** on validation perplexity, but that candidate also caused the largest English regression. Adding downstream English retention to the selection criterion changed the choice to **1e-5**. This is a concrete reason to make the Spark LR screen multi-objective rather than select the lowest Indic loss. [8]

Gnani also reports a tokenizer-stage bug: inherited embedding rows moved despite zero gradients because AdamW's decoupled decay still updated them. The fix snapshots and restores inherited rows. That is relevant to optimizer correctness, but Spark does not need the new-token stage because its vocabulary must remain unchanged. [8]

Sarvam-1 is a **2B model trained from scratch**, not a small-model CPT recipe. Sarvam-M applies SFT and RLVR to **24B Mistral Small**, also not the requested CPT setting. Its tokenizer experiments are nevertheless cautionary: extension damaged retained knowledge, while tokenizer transplant/distillation did not outperform the simpler SFT approach and required additional recovery training. Neither article supplies a transferable Spark CPT LR. [34][33]

Krutrim offers explicit advice: start around **25% original / 75% new data**, adjust for forgetting, and resume at the **last pretraining LR**. It does not publish the complete numeric CPT recipe requested here. This is a competing strategy to aggressive rewarming, not evidence that one universal schedule exists. [6]

**Decision:** Keep the stipulated 50% replay, but select LR using English retention as a constraint. Do not borrow tokenizer expansion, frozen-block training or post-training hyperparameters to fill gaps in Spark's CPT recipe.


## 2. Dense Adaptation Recipes Beyond Indic

### Sailor and Typhoon provide the closest small-model comparisons

Sailor deliberately keeps Qwen1.5's vocabulary after unsuccessful vocabulary-expansion exploration. Its 1.8B model is especially relevant in scale: the released recipe uses a constant **1e-4** LR after warmup, a **4096** context and substantial English/Chinese replay. This makes 1e-4 a defensible upper screening candidate, not a safe default for Spark. [19]

Typhoon 2 supplies a different useful match: unchanged tokenizer, full-parameter training for models up to 8B, and **50% English** CPT replay. Its LR is optimized per model but the values are not published. The report's **3:7 Thai:English** finding belongs to SFT, not CPT, so it must not overwrite the 50/50 CPT recipe. [45]

### Run identity, consumption and vocabulary

| Run | Base size / resulting size | CPT tokens | Target / replay mixture | Full vs LoRA | Vocabulary changed? |
|---|---|---|---|---|---|
| Swallow-7B | Llama-2 7B -> 7B | 100B | 90% Japanese / 10% English | Full-parameter CPT | Yes, +11176 -> 43176 [30] |
| Sailor-1.8B | Qwen1.5 1.8B | About 200B | 140B SEA / 60B English-Chinese and other replay | Exact scope NR | No [19] |
| Sailor-4B | Qwen1.5 4B | About 200B | 140B SEA / 60B replay | Exact scope NR | No [19] |
| Sailor-7B | Qwen1.5 7B | About 200B | 140B SEA / 60B replay | Exact scope NR | No [19] |
| Sailor2-1B | Qwen2.5 0.5B -> 1B, block expansion | Abstract 500B; stage tables sum to 510B | Abstract 400B SEA / 100B replay; stage-2 mixture differs | Exact scope NR; architecture expanded | NR [46] |
| Sailor2-8B | Qwen2.5 7B -> 8B, block expansion | Same 500B versus 510B discrepancy | Same abstract allocation; stage-2 mixture differs | Exact scope NR; architecture expanded | NR [46] |
| SEA-LION v3 Llama | Llama-3.1 8B-Instruct | 200B | 55% SEA / 25% English / 20% code | Exact scope NR | NR [39] |
| SEA-LION v3 Gemma | Gemma-2 9B | 200B | 55% SEA / 25% English / 20% code | Exact scope NR | NR [13][39] |
| Krikri-8B | Llama-3.1 8B Base | About 110B after upsampling | 60% Greek / 22.9% English / 8% parallel / 9.1% math-code | Full after embedding-only initialization | Yes, +20992, resulting 149248 [21] |
| DictaLM 2 | Mistral-7B-v0.1 | About 200B | 50% Hebrew / 50% English | Exact CPT scope NR | Yes, +1000 Hebrew tokens [22] |
| ALLaM-7B | Llama-2 7B | 1.2T | 45% Arabic, half natural and half translated / 55% English | Exact scope NR | Yes, 32000 -> 61568 [25] |
| Fanar Prime | Gemma-2 9B -> 8.78B after vocabulary pruning | 650B | 45% Arabic / 45% English / 10% code | Exact scope NR | Yes, pruned to 128256 [24] |
| Bielik 11B v2 | Mistral-7B-v0.2 -> 11B, depth upscaling | 198B corpus; paper says two epochs; total token-presentations ambiguous | Polish plus English; exact final ratio NR | Exact scope NR | Yes, 32000 -> 32128 [26] |
| Occiglot-7B-EU5 | Mistral-7B-v0.1 | 293B | Approximately 20% each German/Spanish/French/Italian, 13% English, 5% code; rounded figures | Exact scope NR | No [20] |
| TowerBase-7B | Llama-2 7B | 20B dataset; separate consumed total NR | 10 languages; 2/3 monolingual, 1/3 parallel; English replay share NR | Exact scope NR | No [23] |
| Typhoon2-Llama-1B | Llama-3.2 1B | NR; corpus was subsampled | 50% Thai / 50% English | Full | No [45] |
| Typhoon2-Llama-3B | Llama-3.2 3B | NR | 50% Thai / 50% English | Full | No [45] |
| Typhoon2-Qwen-7B | Qwen2.5 7B | NR | 50% Thai / 50% English | Full | No [45] |
| Typhoon2-Llama-8B | Llama-3.1 8B | NR | 50% Thai / 50% English | Full | No [45] |

**Takeaway:** The closest matches are Sailor-1.8B and Typhoon's 1B/3B models, not merely the projects with the most complete tables. Tokenizer preservation is demonstrably viable; it does not require importing vocabulary surgery from other successful adaptations.

### Learning-rate schedules, batches and sequence lengths

| Run | Published LR | Scheduler | Warmup | Final LR | Global batch, tokens/update | Sequence length |
|---|---|---|---|---|---|---|
| Swallow-7B | Peak 1e-4 | Cosine | 1000 steps | Peak/30; about 3.33e-6, derived | 4194304, derived from 1024 sequences | 4096 [30] |
| Sailor-1.8B | 1e-4 | Constant after warmup | 500 steps | 1e-4 | 4M | 4096 [19] |
| Sailor-4B | 1e-4 | Constant after warmup | 500 steps | 1e-4 | 4M | 4096 [19] |
| Sailor-7B | 1e-4 | Constant after warmup | 500 steps | 1e-4 | 4M | 4096 [19] |
| Sailor2-1B | Stage 1: 1e-4; stage 2: 1e-5 | Exact shape NR | NR | NR; stage-2 LR is not a stated terminal LR | Tokens NR; 1024 then 4096 sequences | NR [46] |
| Sailor2-8B | Stage 1: 1e-4; stage 2: 1e-5 | Exact shape NR | NR | NR | Tokens NR; 1024 then 4096 sequences | NR [46] |
| SEA-LION v3 Llama | 1e-5 | Warmup-Stable-Decay, WSD | 10% | 1e-7; decay occupies final 10% | NR | NR [39] |
| SEA-LION v3 Gemma | 1e-5 | WSD | 10% | 1e-7; final 10% decay | NR | NR [39] |
| Krikri-8B | NR | Cosine | NR | NR | NR | 128K, as reported [21] |
| DictaLM 2 | Initial 5e-6 | Cosine | 1000 steps | NR | About 1.5M | 4096 [22] |
| ALLaM-7B | Uses pretrained model's terminal LR, usually 3e-5; not explicitly called a CPT peak | Default shape NR | NR | NR | NR | NR [25] |
| Fanar Prime | Peak 8e-6 | Cosine | 100 steps | 1e-6 | About 4.4M; 1071 sequences | 4096 [24] |
| Bielik 11B v2 | 2e-5 | Cosine | 50 iterations | 9e-6 | Up to 4194304, derived from 128 full-length sequences | 32768 [26] |
| Occiglot-7B-EU5 | 1e-5 | Cosine annealing with warmup | 420 steps | NR | 4194304, derived | 8192 [20] |
| TowerBase-7B | 3e-5 | Cosine | NR | 3e-6 | 1.57M | NR [23] |
| Typhoon2-Llama-1B | Optimized individually; value NR | Cosine | NR | NR | NR | 8192 [45] |
| Typhoon2-Llama-3B | Optimized individually; value NR | Cosine | NR | NR | NR | 8192 [45] |
| Typhoon2-Qwen-7B | Optimized individually; value NR | Cosine | NR | NR | NR | 8192 [45] |
| Typhoon2-Llama-8B | Optimized individually; value NR | Cosine | NR | NR | NR | 8192 [45] |

The terminal LR varies substantially even among successful cosine recipes. Neither cosine-to-zero nor a particular warmup fraction is a universal CPT convention. The Sailor2 abstract/stage-table discrepancy and Bielik's corpus/epoch ambiguity should remain visible rather than be silently normalized.

### Optimizer and embedding policies

| Run | Optimizer | Adam betas | Weight decay | Gradient clip | Embedding LR | Embedding freeze during main CPT |
|---|---|---|---|---|---|---|
| Swallow-7B | AdamW, epsilon 1e-8 | 0.9, 0.95 | 0.1 | 1.0 | NR | No separate freeze reported; full CPT [30] |
| Sailor-1.8B | AdamW | 0.9, 0.95 | 0.1 | 1.0 | NR | NR [19] |
| Sailor-4B | AdamW | 0.9, 0.95 | 0.1 | 1.0 | NR | NR [19] |
| Sailor-7B | AdamW | 0.9, 0.95 | 0.1 | 1.0 | NR | NR [19] |
| Sailor2-1B | NR | NR | NR | NR | NR | NR [46] |
| Sailor2-8B | NR | NR | NR | NR | NR | NR [46] |
| SEA-LION v3 Llama | NR | NR | NR | NR | NR | NR [39] |
| SEA-LION v3 Gemma | NR | NR | NR | NR | NR | NR [39] |
| Krikri-8B | Numeric optimizer recipe NR | NR | Used; value NR | Used; value NR | CPT LR NR; separate new-token stage | No, all parameters unfrozen after initialization [21] |
| DictaLM 2 | Numeric optimizer type NR in extracted recipe | NR | 0.1 | NR | CPT value NR; earlier new-token distillation 5e-6 | CPT policy NR; earlier only new-token embeddings trained [22] |
| ALLaM-7B | NR | NR | NR | NR | NR | NR [25] |
| Fanar Prime | AdamW, epsilon 1e-8 | 0.9, 0.95 | 0.01 | 1.0 | NR | NR [24] |
| Bielik 11B v2 | AdamW | 0.9, 0.95 | 0.1 | 1.0 | NR | NR [26] |
| Occiglot-7B-EU5 | AdamW | NR | NR | NR | NR | NR [20] |
| TowerBase-7B | NR | NR | NR | NR | NR | NR [23] |
| Typhoon2-Llama-1B | AdamW | NR | NR | 1.0 | NR | NR separately; full training [45] |
| Typhoon2-Llama-3B | AdamW | NR | NR | 1.0 | NR | NR separately; full training [45] |
| Typhoon2-Qwen-7B | AdamW | NR | NR | 1.0 | NR | NR separately; full training [45] |
| Typhoon2-Llama-8B | AdamW | NR | NR | 1.0 | NR | NR separately; full training [45] |

**Takeaway:** AdamW betas **0.9/0.95**, clipping **1.0** and matrix decay **0.1** have repeated primary-source support, but embedding-specific policies are usually undisclosed. Repetition across other models supports a starting baseline, not a Spark-specific optimum.

### SEA-LION v4, Fanar and size exclusions

SEA-LION paper revision **v4** is not the same thing as model family **SEA-LION v4**. The actual Apertus-SEA-LION-v4-8B documentation describes instruction tuning and model merging, not a disclosed CPT run. Its CPT tokens, mixture, LR, scheduler, warmup, terminal LR, batch, sequence, betas, decay, clipping, embedding policy and vocabulary-change policy are therefore NR for CPT. [39][44]

The actual Gemma-SEA-LION-v4-27B is outside the requested size band. It reports **500B CPT tokens**, **8k** training lengths, decoupled AdamW, cosine annealing, LR **4e-8**, and global batch **1024 sequences**; a nominal approximately 8M-token batch is derived, not directly stated. Exact target/replay share, full-versus-LoRA scope, warmup, final LR, betas, decay, clipping, embedding policy and vocabulary modification are NR. This unusually small LR should not become a Spark default. [47]

Fanar **Prime** is the CPT case; Fanar **Star** is trained from scratch. Bielik's named 11B v2 and Nanda's final 10B likewise sit outside a strict final-size 1B-9B comparison even though their starting checkpoints are relevant. [24][26][7]

**Decision:** Transfer optimizer conventions primarily from unchanged, dense models. Treat expansion, pruning, instruction-checkpoint adaptation and out-of-band models as contextual evidence, not matched experiments.


## 3. Schedule Theory: A Fair Screen Before A Long Run

### Rewarming is useful, but the optimal amount is not universal

Gupta et al. study Pythia-410M moving from the Pile to SlimPajama, without replay. They test peaks **1.5e-4, 3e-4 and 6e-4**. Higher peaks improve final downstream loss but worsen upstream loss; rewarming can initially worsen both. Their warmup-length comparisons do not establish a generally optimal warmup fraction. [16]

Ibrahim et al. extend rewarm-plus-redecay experiments to approximately 405M and 10B models and weak and stronger shifts. They test **0.5x, 1x and 2x** the original peak and emphasize the adaptation/forgetting trade-off. Their main replay experiments rewarm to **1x**, rather than proclaiming a universally correct fraction. [43]

**My conditional relative recommendation is 0.5 x the original pretraining peak**, using the retention-favoring tested arm as a conservative starting target. This is an engineering extrapolation, not an Ibrahim theorem. Since Spark's original peak is unpublished, no absolute number can honestly be certified as that fraction.

There is substantial counterevidence against blindly rewarming to the original peak. Parmar et al.'s 15B study uses CPT peak **4.5e-5**, only **0.1x** its original **4.5e-4**, favors no warmup, and selects a terminal LR of **4.5e-7**, or 1/100 of the CPT peak. ALLaM instead resumes using the pretrained model's terminal LR, usually **3e-5**, and reports limited success with increasing/decaying alternatives. These are different starting checkpoints and distribution shifts, not interchangeable validations of one schedule. [10][25]

### What the other named theories actually establish

| Source / framework | Finding relevant to Spark | What it does not establish |
|---|---|---|
| D-CPT Law, Que et al., 2024 | Models loss as a function of model size, token budget and domain mixture; its fixed-mixture limit is consistent with conventional scaling laws | No universal LR, no proof that 50% replay is optimal, and no Spark-calibrated forecast without fitting [29] |
| Yildiz, Ravichandran et al., 2024 | Smaller models in the study learn and forget faster; size matters to continual adaptation | Not a validated LR scaling formula for this 1.7B checkpoint; their automatic LR/dropout setup is not a strong transferable CPT recipe [14][28] |
| Stability-gap paper, Guo et al., 2025 | Downstream task performance can fall and recover while domain perplexity steadily improves | An early target-task dip does not alone prove irreversible forgetting, nor does falling PPL prove capability improvement [11] |
| Dagan et al., 2024 tokenizer study | Tokenizer replacement can recover with substantial adaptation; some 1.5B experiments become competitive around 50B tokens | No reason to replace Spark's fixed tokenizer or borrow swap-specific settings [2] |
| ZeTT, 2024 | A trained hypernetwork predicts embeddings for new tokenizers | Not a function-preserving tokenizer swap and not needed for unchanged-vocabulary CPT [27] |

**Takeaway:** These works explain the risks and experimental design better than they determine Spark's LR. In particular, a scaling law for mixture and tokens cannot fill an unpublished optimizer field.

### Run the three-point screen on one complete horizon

Use identical initialization, token stream, token batch, sequence length, optimizer groups, warmup budget and complete decay horizon across all LR arms. Change only peak LR and its proportionate terminal LR. These controls are engineering recommendations motivated by the published adaptation/forgetting comparisons.

For a token horizon `T`, warmup `W`, peak `p` and terminal fraction `a`, define the post-warmup schedule as:

`lr(t) = a*p + 0.5*(1-a)*p*(1 + cos(pi*(t-W)/(T-W)))`.

The formula is the ordinary cosine schedule written in token units. Fitting its endpoint to the training budget follows Ibrahim; the proposed terminal fraction **a = 0.1** transfers its listed cosine configurations. [43]

Do not rank a fully cooled short run against an early checkpoint of a long cosine, and do not present a cooled pilot as the unchanged prefix of the long schedule. At the same token count, those models have experienced different LRs and different integrated updates. Extending the cooled pilot means defining another training phase, not recovering the missing long-horizon trajectory.

WSD is an alternative if reusable prefixes are essential: SEA-LION v3 publishes **10% warmup, 80% stable and 10% decay**. A stable trunk with separately cooled evaluation branches can avoid repeatedly cooling the training trunk. That branching policy is an engineering extrapolation; it is not the same experiment as the proposed complete-cosine screen. [39]

**Decision:** Use complete short cosine runs for an interpretable first screen. Select by an Indic-versus-retention Pareto frontier, then rebuild the schedule for the separate long horizon; do not claim that a short-run winner is proven optimal at 100B tokens.


## 4. Regularizers: Preserve The Starting Function And Checkpoint Lineage

A crucial distinction is **changing the forward function before training** versus **changing only the training objective or update rule**. Every successful CPT run changes weights and therefore its eventual function; that is intended. The avoidable risk is inserting a new normalization, cap or tokenizer and silently ceasing to start from the released model's function.

### What primary sources actually support

| Intervention | Primary-source use and numbers | Immediate effect on the loaded model | Recommendation for Spark CPT |
|---|---|---|---|
| z-loss | OLMo 2 uses log-partition regularization; its prose gives 1e-4 times log-squared partition, while its comparison table lists weight 1e-5 | Auxiliary loss; no necessary inference-graph change, but gradients change | Off initially. The report's coefficient discrepancy is unresolved; do not invent a Spark value [42] |
| Embedding weight decay 0 versus 0.1 | OLMo 2 removes embedding decay after observing small norms and unstable dynamics with standard 0.1 decay | Update-rule change, not a forward-graph mutation | Prefer 0 on Spark's tied embedding tensor; this is a transfer from pretraining evidence, not an Indic-CPT ablation [42] |
| QK-norm | OLMo 2 applies RMSNorm to query/key projections; normalization changes are studied jointly | Changes attention scores immediately if inserted into an existing checkpoint | Do not add it to Spark's released architecture [42] |
| Gemma logit soft-cap | Gemma 2 uses `c*tanh(logit/c)`, with attention cap 50 and final-logit cap 30 | Changes attention and output probabilities immediately | Do not graft these caps onto Spark; they belong to Gemma's architecture [38] |
| NEFTune | Instruction-fine-tuning embedding noise; tested alpha values include 5, 10 and 15 | Changes training forward passes; ordinarily disabled at inference | Not a demonstrated CPT retention control; leave off [37] |
| max-z | Baichuan 2 uses `2e-4 * z^2`, where z is the maximum logit, in scratch pretraining | Auxiliary training loss, not an architecture change | Leave off unless diagnosing logit instability; no matched Spark/Indic coefficient [49] |
| EMA | 2026 Anytime Pretraining uses horizon-independent averaging; experiments cover 150M/300M C4 models | Shadow EMA leaves raw weights untouched; selecting EMA changes the exported checkpoint | Optional diagnostic shadow only; no validated Spark averaging coefficient [50] |
| Base/CPT interpolation; WiSE-FT-style blending; soups | 2024 Branch-and-Merge language adaptation studies linear, SLERP and Model Stock merging; Gemma averages post-training models | Changes endpoint weights and function, although tensor shapes can remain compatible | Keep raw CPT as the intended parent; evaluate any merge as a separately identified sibling [36][38] |
| Dropout inserted during CPT | ALLaM reports Arabic help but adverse retention implications when adding dropout to a pretrained model | Changes training computation; inference may remain unchanged when disabled | Do not add it as an unvalidated default [25] |

**Takeaway:** The best-supported low-disruption intervention here is parameter-grouped decay. Architecture-level stabilizers from strong scratch models are not automatically safe checkpoint adaptations.

### OLMo 2: stability evidence is not a matched retention experiment

OLMo 2's embedding-decay case is useful because it identifies a concrete mechanism: decay can shrink embeddings into an unhealthy norm regime, while excluding them allows healthier norms. It is stronger evidence than a generic claim that more regularization prevents forgetting. However, the model family and training history differ from Spark; it does not establish a measured English-retention benefit for this CPT task. [42]

Spark's tied embeddings make the grouping consequential. The input embedding and output head share one parameter tensor under the supplied specification. Assign it to one zero-decay group only; do not let an output-head grouping add a second update or contradictory decay. Keep that tensor trainable at the global LR. This grouping and equal embedding LR are explicit engineering choices, not published Spark CPT settings.

### Branch-and-Merge: useful evidence, not a free parent-preserving operation

Branch-and-Merge adapts Mistral-7B and Llama-3-8B to German and Bulgarian using branching and model merging. Its Bulgarian results report less English deterioration than sequential CPT, demonstrating that weight-space operations can improve the adaptation/retention compromise. But the method is a structured training procedure, not proof that averaging a base and final CPT checkpoint at an arbitrary coefficient will work. [36]

For compatible tensors, a base/CPT interpolation can be written `theta_mix = (1-c)*theta_base + c*theta_CPT`. The published merging formulation supports this operation, but supplies no universal Spark value for `c`. A WiSE-FT-style or model-soup experiment is therefore optional model selection, not a default regularizer. [36]

Ordinary z-loss, max-z, decay changes and training-only noise do not inherently block later SFT or require inference changes. QK-norm, soft-capping, NormHead-style normalization, vocabulary replacement and embedding untying do change the starting model or its interface. EMA and interpolation do not inherently block SFT either, but selecting them changes which checkpoint is its parent. Preserve raw CPT checkpoints, exact configuration and tensor tying; do not describe a merged checkpoint as the unmodified CPT endpoint.

**Decision:** Start without auxiliary losses, embedding noise, new normalization, output caps or weight averaging. Add a single intervention only to solve an observed problem, and retain the unmodified-architecture raw CPT lineage.


## 5. Forgetting Controls: Select On Retention, Not Mixture Loss

### Fifty percent replay is evidence-backed, not a guarantee

The closest direct cases are Hindi Mini's equal Hindi/English allocation, Nanda's selected 1:1 mix and Typhoon 2's 50% English CPT. DictaLM 2 independently uses 50% Hebrew and 50% English. These are useful precedents for the user's fixed mixture, but differ in capacity, tokenizer treatment, LR and budget. [31][7][45][22]

Ibrahim's replay study keeps compute fixed: replay replaces new-domain tokens rather than adding free training. The paper selects **5% replay** for its weak shift and **25%** for its stronger shift because these achieve a favorable compromise while exposing more new tokens. Its 50% variants can achieve good joint losses while processing substantially fewer new-domain tokens. Thus 50% is a reasonable stipulated constraint, not a literature-wide optimum. [43]

At a 100B-token budget, the fixed mixture provides **50B Indic and 50B replay tokens**, by arithmetic. A lower combined validation loss can still conceal English deterioration: the Indic improvement may simply dominate the weighted average. Log separate losses and make retention an explicit selection constraint.

### How much English degradation have small projects actually accepted?

Hindi Mini offers a base-to-base comparison in the requested approximate size range. The authors acknowledge English degradation while describing the results as competitive; they do not publish a predeclared maximum acceptable drop. [31]

| English benchmark | Original Mini base | Hindi-CPT base | Change, percentage points, derived |
|---|---:|---:|---:|
| MMLU | 58.60 | 56.37 | -2.23 |
| ARC-Challenge | 50.90 | 46.08 | -4.82 |
| HellaSwag | 75.00 | 74.64 | -0.36 |
| TruthfulQA MC2 | 42.72 | 41.05 | -1.67 |

Source: the paper's base-model table, not its later instruction-tuned results. [31]

**Takeaway:** An apparently acceptable aggregate model can lose several points on a specific English task even with equal replay. These observations do not authorize a 2-point or 5-point Spark acceptance threshold.

For Sailor's 1.8B/4B and Typhoon 2's 1B/3B runs, the located primary evaluations do not supply a numeric English-drop tolerance. Typhoon's Thai exam tables are not English retention measurements. At larger sizes, Swallow-7B's reported English average falls **49.0 -> 44.0**, and Krikri's pure-CPT English average falls **66.2 -> 62.2** before subsequent annealing stages. These are contextual trade-offs, not matched 1B-4B tolerances. [19][45][30][21]

### Optimizer and loop controls to implement

1. **Constrain peak LR before adding machinery.** Prefer the lowest LR on the feasible Indic/English frontier, not the candidate with best Indic PPL alone. Gnani's 1e-4-to-1e-5 selection reversal is the concrete motivating case. [8]
2. **Reset unavailable optimizer state deliberately.** Starting fresh AdamW states is consistent with Ibrahim's open-weight CPT setup. Preserve optimizer and scheduler state for ordinary resumes; do not repeatedly reset them at arbitrary checkpoints. The latter is an engineering recommendation. [43]
3. **Track domain-specific losses.** Maintain fixed English, math/code and per-Indic-language held-outs, with separate native-script and romanized subsets where relevant. Keep evaluation tokenization, packing and lengths fixed across checkpoints. This monitoring design is an engineering extrapolation, not a published numeric recipe.
4. **Separate task regression from the stability gap.** The stability-gap paper reports a task-performance dip while domain PPL improves; OpenLLaMA-3B's early decline can last through the first 5B tokens. A short Spark pilot might therefore be inconclusive on some tasks. Do not automatically terminate because the first downstream score falls. [11]
5. **Use predeclared stop conditions.** Stop for non-finite loss/gradients or genuine divergence. For capability retention, require the project's predeclared English limits and uncertainty estimates; no located primary source establishes universal Spark thresholds or a universal patience interval. Keep intermediate checkpoints and permit an inconclusive pilot outcome rather than forcing a winner.

NPI's 2026 English results reinforce this caution: some code metrics improve, but six metrics remain below baseline across all three Indic allocations, and each setting has only one training run. Its broad success statement must not be translated into a guarantee of no forgetting. [9]

**Decision:** Use replay as the fixed exposure constraint and LR/checkpoint selection as the main optimizer-loop retention controls. Never infer retention from target loss alone or treat an early V-shaped task dip as a universal stop signal.


## 6. What XHToken Actually Publishes For Spark

The Spark-X2.5 repository and Spark-X2.6 card recommend XHToken's LlamaFactory fork for fine-tuning. The located X2.6 card explicitly describes a **post-trained** checkpoint, so it must not be confused with the base checkpoint specified in the question. Neither source provides the original pretraining peak LR or a complete CPT recipe. [3][4]

The fork does contain model-specific material under `examples/torch_spark2_5`: full-SFT YAML files for 1.7B and 4B and a fine-tuning guide. These are materially better evidence than generic upstream examples, but they remain SFT examples. [48][51][52]

### Released Spark-specific example settings

| Example | Training scope / optimizer flag | LR | Schedule / final LR | Warmup | Batch | Sequence | Other numeric settings |
|---|---|---|---|---|---|---|---|
| 1.7B full-SFT YAML | Full; `use_muon: true` | 4e-5 | `cosine_with_min_lr`; minimum rate 0.1, giving 4e-6, derived | 0 | Microbatch 4, accumulation 4; world size NR | 2048 | 3 epochs; BF16; WD 0.1; Adam fields 0.9/0.95, epsilon 1e-8 [51] |
| 4B full-SFT YAML | Full; `use_muon: true` | 4e-5 | Same scheduler and minimum rate | 0 | Microbatch 4, accumulation 4; world size NR | 2048 | 3 epochs; BF16; WD 0.1; Adam fields 0.9/0.95, epsilon 1e-8 [52] |
| 1.7B full-SFT guide | Full; Muon enabled | 4e-5 | Cosine with explicit minimum 8e-6 | 0 | Microbatch 4, accumulation 4; guide launch uses 8 processes | 2048 | 3 epochs; WD 0.1; Adam fields 0.9/0.95, epsilon 1e-8 [53] |
| 4B full-SFT guide | Full; Muon enabled | 4e-5 | Cosine | 0 | Microbatch 1, accumulation 4; 8-process launch | 2048 | 3 epochs; WD 0.1; Adam fields 0.9/0.95, epsilon 1e-8 [53] |

**Takeaway:** The YAML and guide are not identical, notably on the 1.7B terminal LR and the 4B microbatch. Pin the exact file revision instead of treating the repository as one immutable recipe.

With the guide's 8-process launch, the 1.7B full-SFT example has **128 sequences/update**, or at most **262144 tokens** at the stated cutoff; both numbers are derived. Actual supervised tokens can be lower because of example lengths and masking. The guide's LoRA examples instead use **5e-5**, **one epoch** and zero warmup; those values are not evidence for full-parameter CPT. [53]

The `use_muon` flag is important: the presence of Adam betas in the YAML does not establish that every matrix is updated by ordinary AdamW. Do not copy these examples and describe the result as a published pure-AdamW CPT recipe. Gradient clipping, embedding-specific LR/decay, vocabulary-change policy and a Spark CPT budget are not specified by these SFT examples. [51][52]

For this project, use the model-compatible implementation but configure next-token pretraining rather than the guide's chat-SFT stage and template-driven masking. Audit causal labels, sliding-window masks, partial RoPE, sigmoid gates, packing and tied-weight handling against the released model before training. These are engineering requirements arising from the supplied fixed architecture, not newly claimed XHToken hyperparameters.

**Decision:** There is genuine official Spark fine-tuning guidance, but no located primary-source basis for calling **4e-5** the original pretraining LR, assuming zero warmup is safe for CPT, or replacing Spark's attention implementation with a generic Llama path.


## 7. Recommended Pilot And Separate 100B Recipe

**These are proposed transfer recipes, not published Spark recipes.** Labels below distinguish **F** = fixed by the question, **P** = published elsewhere, **D** = arithmetic derivation, and **X** = engineering extrapolation to Spark. A published value becomes a transfer assumption when applied to this checkpoint.

### Three-point pilot: change only peak and proportional terminal LR

| Arm | Peak LR | Terminal LR | Primary-source anchor | Status |
|---|---:|---:|---|---|
| Retention-first | 1e-5 | 1e-6 | Gnani's selected peak; also Occiglot and SEA-LION v3 [8][20][39] | Peak P; Spark use X; terminal D from 0.1 ratio |
| Middle | 3e-5 | 3e-6 | Tower's published start/end pair [23] | P; Spark use X |
| Adaptation-first | 1e-4 | 1e-5 | Sailor, including 1.8B, and Swallow-7B peaks [19][30] | Peak P; Spark use X; terminal D from 0.1 ratio |

**Takeaway:** This is deliberately not described as 0.5x/1x/2x Spark's original LR. That conversion is impossible with the current disclosure. The highest arm is a screened risk, not the recommended production default.

I recommend a **3B-token horizon per arm**, an **X** budget choice: **9B aggregate screening tokens**, calculated across the three arms. If "a few billion" is a hard aggregate cap of 3B, use **1B per arm**, also X, and treat the result as a preliminary stability screen rather than a reliable long-run ranking. The stability-gap evidence makes that limitation substantive. [11]

### Complete training settings and provenance

| Setting | Pilot recommendation | Separate 100B recommendation | Provenance / qualification |
|---|---|---|---|
| Initialization | Released Spark base, independently for each arm | Restart from the same released base using the selected recipe | X experimental control; preserves a clean comparison |
| Parameters updated | All parameters, including tied embeddings and existing gates | Same | F: full-parameter CPT; no LoRA or frozen-backbone substitution |
| Token budget | 3B per arm; 9B total | 100B | Pilot X; long-run F |
| Target/replay | 50% Indic / 50% replay; 1.5B each per 3B arm | 50B Indic / 50B replay | F proportions; counts D; matched proportions in Hindi Mini and Typhoon 2 [31][45] |
| Sequence length | 4096; use 2048 only if selected before the screen | Same chosen length | Both F options; selecting 4096 is X; Sailor/Swallow provide 4096 precedents [19][30] |
| Global batch | 2260992 non-padding training tokens/update | Same | D: 1104 x 2048 from Ibrahim's listed setup; transfer X [43] |
| Packed sequences/update | 552 at 4096, or 1104 at 2048 | Same | D from the token batch; accumulation and device microbatch depend on hardware |
| Optimizer | AdamW, not the fork's Muon option | Same | P in Gupta, Swallow and Sailor; transfer X [16][30][19] |
| Adam betas | 0.9, 0.95 | Same | P in those CPT sources; transfer X [16][30] |
| Adam epsilon | 1e-8 | Same | P in Gupta/Swallow; transfer X [16][30] |
| Gradient norm clip | 1.0 | Same | P in Gupta/Swallow/Typhoon; transfer X [16][30][45] |
| Peak LR | Three arms above | Retention-constrained winner; do not raise automatically | Absolute grid X; no source establishes the 100B optimum |
| Warmup | 100 optimizer updates, equal to 226099200 tokens at this batch | 1% of horizon = 1B tokens | Pilot 100-step precedent: Fanar Prime; converting to tokens D and applying here X. Long 1% precedent: Ibrahim's listed cosine configuration; transfer X [24][43] |
| Warmup shape | Linear to peak | Linear to peak | P in Ibrahim; transfer X [43] |
| Decay | Cosine after warmup, reaching 0.1 x peak at the shared pilot endpoint | Recomputed cosine reaching 0.1 x peak at 100B | P shape/ratio in Ibrahim; horizon-specific application X [43] |
| Embedding LR | Same as global LR; no freeze | Same | X; no located primary Spark CPT embedding multiplier |
| Additional regularizers | No added z-loss, max-z, NEFTune, QK-norm, soft-cap, EMA replacement or interpolation | Same unless a separately tested problem warrants a change | X conservative baseline, not a claim that these methods never help |
| Vocabulary and graph | Unchanged vocabulary, tying, gates, masks and RoPE | Same | F; no context-extension phase |

The pilot's **100-step warmup** is intentionally not presented as the standard 1% convention: at the proposed batch it occupies approximately **226M tokens**, or **7.54%** of a 3B arm, both derived. A 1% pilot warmup would last only approximately **13 updates**; choosing the longer published step-count precedent is an explicit engineering judgment. The long warmup is approximately **442 updates**, derived, and should be implemented by actual token count rather than a rounded step percentage.

Do not change the token batch when switching from 2048 to 4096 and then attribute the difference solely to context length. If hardware cannot support this batch economically, choose a smaller batch before the screen and recompute all token-to-step conversions; the literature does not establish a Spark-specific microbatch or accumulation count.

### Parameter groups: zero decay is not freezing

| Parameter group | Weight decay | LR / trainability | Evidence and status |
|---|---:|---|---|
| Existing attention, MLP and gate projection matrices | 0.1 | Global LR; trainable | P matrix-decay baseline in Gupta, Swallow and Sailor; applying to Spark's existing gates is X [16][30][19] |
| Single tied input-embedding/output-head tensor | 0 | Global LR; trainable; include once | P embedding-decay exclusion in OLMo 2; tied-tensor application X [42] |
| Normalization scales and any biases | 0 | Global LR; trainable | X grouping choice; no located primary Spark CPT value |

**Takeaway:** This baseline changes the optimizer grouping without changing the loaded forward graph. No parameter is frozen, and no unsupported embedding LR multiplier is introduced.

Initialize fresh optimizer states at the first CPT step; preserve them on true resumes. Save the raw model, optimizer, scheduler, RNG/data-position state and complete configuration together. Select the final checkpoint under predeclared English constraints, not necessarily the last one. These operational recommendations complement Ibrahim's demonstrated optimizer-reset setup; they are not claimed as a published Spark recipe. [43]

**Decision:** Run the middle arm as the initial operational baseline, but promote only a retention-feasible result. If all arms violate the acceptance criteria, do not commit 100B tokens merely because one has the best Indic perplexity.


## Synthesis

### Three strategies solve different problems

| Strategy | Mechanism | Scope and evidence | Trade-off | Time-horizon implication |
|---|---|---|---|---|
| Conservative terminal-LR continuation | Limit update magnitude around a mature checkpoint | Krutrim advice; ALLaM terminal-LR continuation; Parmar's 0.1x-peak 15B recipe [6][25][10] | Stronger retention bias can under-adapt a small model | May look attractive early without proving enough language acquisition at 100B |
| Rewarm plus redecay with replay | Restore plasticity, then cool to consolidate | Gupta and Ibrahim controlled experiments; full-parameter Typhoon replay precedent [16][43][45] | More learning can mean more forgetting; replay consumes the same finite budget | Requires an explicit horizon or a deliberately chosen horizon-free alternative |
| Structural adaptation or weight merging | Add capacity, alter tokenization, or shrink/reshape task updates through merging | Nanda frozen-backbone expansion, Sailor2 depth expansion, and Branch-and-Merge [7][46][36] | Different architecture or checkpoint parent; extra selection complexity | Successful long adaptations do not validate a drop-in Spark optimizer setting |

**Takeaway:** Spark's constraints favor the second strategy with a conservative LR screen and elements of the first, while excluding structural adaptation. Merging remains an optional sibling experiment, not a requirement or a substitute for retained replay capabilities.

The non-obvious tension is that stronger scratch-model stability mechanisms can be worse engineering choices for checkpoint-preserving adaptation. QK-norm and Gemma caps are legitimate architectural choices in their own models, but installing them in Spark changes the initial function before its first CPT update. Conversely, embedding-decay exclusion changes only the update rule and is therefore a less disruptive experiment. [42][38]

Another tension is that the best short-horizon target-loss model may be the wrong long-run parent. Gnani's selection reversal and the stability-gap observations show two different failure modes: target loss can favor excessive forgetting, while early task scores can reject a model that would later recover. The answer is not a fabricated patience number; it is separate loss/task measurements, a complete shared screen schedule and permission to declare the pilot inconclusive. [8][11]

### Remaining uncertainties

- **Original Spark peak and terminal pretraining LR:** unpublished in the located primary materials. The proposed absolute grid cannot be mapped honestly to the recommended relative starting target.
- **Spark CPT optimizer details:** original betas, decay groups, clipping, embedding LR and optimizer-state availability are not disclosed by the located CPT materials; the official SFT examples do not resolve them. [51][53]
- **Transfer across architectures:** most complete recipes concern 7B-class models with different attention and embedding designs; Sailor-1.8B is closer in scale, not identical in function. [19]
- **Long-horizon ranking:** a few-billion-token screen does not prove the best LR at 100B, especially when task recovery can outlast a short pilot. [11]
- **Replay equivalence:** 50% generic English/code is not necessarily 50% of Spark's original training distribution. Its exact original mixture is not established here.
- **English acceptance limits:** projects report trade-offs, not a universal acceptable drop for 1B-4B models. Hindi Mini's observed losses should inform risk discussion, not become an automatic policy. [31]
- **Long-context retention:** 2048/4096 CPT does not establish retention of the declared 1048576-token capability. Keep the supplied configuration unchanged and evaluate any long-context capability that matters; no source-backed guarantee is available for this run.
- **Published ambiguities:** Nanda's Hindi corpus versus consumed comparison budget, Sailor2's 500B versus 510B accounting, Bielik's 198B/two-epoch wording and OLMo 2's z-loss coefficient discrepancy remain unresolved. [7][46][26][42]

**Bottom line:** Preserve Spark's exact architecture and tokenizer, screen **1e-5 / 3e-5 / 1e-4** under identical complete schedules, use explicit AdamW parameter groups and English retention gates, and rebuild the winner's schedule for the separate **100B-token** run. These numeric recommendations are source-anchored extrapolations, not undisclosed Spark hyperparameters.

## References

1. *Adapting Multilingual LLMs to Low-Resource Languages using Continued Pre-training and Synthetic Corpus: A Case Study for Hindi LLMs - ACL Anthology*. https://aclanthology.org/2025.indonlp-1.6/
2. *Getting the most out of your tokenizer for pre-training and domain adaptation*. https://arxiv.org/html/2402.01035v2
3. *GitHub - XHToken/Spark-X2.5: Spark-x2.5 open model series. Pushing the Limits of Agentic Capabilities in On-Device Models · GitHub*. https://github.com/XHToken/Spark-X2.5
4. *XHToken/Spark-X2.6-1.7B · Hugging Face*. https://huggingface.co/XHToken/Spark-X2.6-1.7B
5. *D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models*. https://arxiv.org/abs/2406.01375
6. *Krutrim LLM: Multilingual Foundational Model for over a Billion People*. https://arxiv.org/html/2502.09642v2
7. *Llama-3-Nanda-10B-Chat:An Open Generative Large Language Model for Hindi*. https://arxiv.org/html/2504.06011
8. *The Cost of Being Understood: LLMs for Indian Languages*. https://www.gnani.ai/resources/research/evon-v3-3-building-sovereign-llms-for-indian-languages
9. *Scaling Synthetic Pre-training Data and Evaluation for Indic LLMs*. https://arxiv.org/pdf/2610.26006v1
10. *Reuse, Don’t Retrain: A Recipe for Continued Pretraining of Language Models*. https://arxiv.org/html/2407.07263v1
11. *Efficient Domain Continual pretraining by Mitigating the Stability Gap*. https://aclanthology.org/2025.acl-long.1578.pdf
12. *Simple and Scalable Strategies to Continually Pre-train Large Language Models*. https://arxiv.org/abs/2403.08763
13. *SEA-LION v3: Advancing Southeast Asian AI in An Open, Inclusive, and Responsible Manner | SEA-LION*. https://sea-lion.ai/blog/sea-lion-v3-2
14. *Investigating Continual Pretraining in Large Language Models: Insights and Implications*. https://arxiv.org/abs/2402.17400
15. *Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities*. https://arxiv.org/abs/2404.17790
16. *Continual Pre-Training of Large Language Models: How to (re)warm your model?*. https://arxiv.org/pdf/2308.04014
17. *Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs*. https://arxiv.org/html/2502.12982v1
18. *Typhoon 2: A Family of Open Text and MultimodalABE.Thai Large Language Models*. https://arxiv.org/html/2412.13702v2
19. *Sailor: Open Language Models for South-East Asia*. https://arxiv.org/html/2404.03608v1
20. *occiglot/occiglot-7b-eu5 · Hugging Face*. https://huggingface.co/occiglot/occiglot-7b-eu5
21. *Krikri: Advancing Open Large Language Models for Greek*. https://aclanthology.org/2025.findings-emnlp.268.pdf
22. *Adapting LLMs to Hebrew: Unveiling DictaLM 2.0 with Enhanced Vocabulary and Instruction Capabilities*. https://arxiv.org/html/2407.07080v1
23. *T : An Open Multilingual Large Language Model for Translation-Related Tasks X*. https://arxiv.org/pdf/2402.17733
24. *Fanar An Arabic-Centric Multimodal Generative AI Platform*. https://alt.qcri.org/~ndurrani/pubs/fanar.pdf
25. *ALL M : L L M A E*. https://proceedings.iclr.cc/paper_files/paper/2025/file/54c15a3033686e7999aecd2740c5a7c4-Paper-Conference.pdf
26. *Bielik 11B v2 Technical Report*. https://arxiv.org/pdf/2505.02410
27. *Zero-Shot Tokenizer Transfer*. https://proceedings.neurips.cc/paper_files/paper/2024/file/532ce4fcf853023c4cf2ac38cbc5d002-Paper-Conference.pdf
28. *Investigating Continual Pretraining in Large Language Models: Insights and Implications*. https://arxiv.org/pdf/2402.17400
29. *D-CPT Law: Domain-specific Continual Pre-Training Scaling Law for Large Language Models*. https://arxiv.org/pdf/2406.01375
30. *Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities A X*. https://arxiv.org/pdf/2404.17790
31. *Adapting Multilingual LLMs to Low-Resource Languages using Continued Pre-training and Synthetic Corpus: A Case Study for Hindi LLMs*. https://aclanthology.org/2025.indonlp-1.6.pdf
32. *GitHub - XHToken/LlamaFactory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024) · GitHub*. https://github.com/XHToken/LlamaFactory
33. *Sarvam-M multilingual language model | Sarvam AI*. https://www.sarvam.ai/blogs/sarvam-m
34. *Sarvam 1: the first Indian language LLM | Sarvam AI*. https://www.sarvam.ai/blogs/sarvam-1
35. *Scaling Indic Pretraining: 2.8 Trillion Tokens and an Open Data Generation Pipeline*. https://huggingface.co/blog/nvidia/scaling-indic-pretraining
36. *Mitigating Catastrophic Forgetting in Language Transfer via Model Merging*. https://aclanthology.org/2024.findings-emnlp.1000.pdf
37. *NEFT : N E I I - F*. https://arxiv.org/pdf/2310.05914v1
38. *Gemma 2: Improving Open Language Models at a Practical Size*. https://arxiv.org/html/2408.00118v3
39. *SEA-LION: Southeast Asian Languages in One Network*. https://arxiv.org/html/2504.05747v4
40. *Output Embedding Centering for Stable LLM Pretraining*. https://arxiv.org/html/2601.02031v1
41. *Data-Constrained Language Model Pretraining: Improved Regularization and Scaling Laws*. https://arxiv.org/pdf/2606.06888
42. *2 OLMo 2 Furious*. https://arxiv.org/html/2501.00656v3
43. *Simple and Scalable Strategies to Continually Pre-train Large Language Models*. https://arxiv.org/pdf/2403.08763
44. *Apertus-SEA-LION-v4-8B*. https://docs.sea-lion.ai/models/sea-lion-v4/apertus-sea-lion-v4-8b
45. *Typhoon 2: A Family of Open Text and Multimodal Thai Large Language Models D X*. https://arxiv.org/pdf/2412.13702
46. *Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs X*. https://arxiv.org/pdf/2502.12982
47. *Gemma-SEA-LION-v4-27B*. https://docs.sea-lion.ai/models/sea-lion-v4/gemma-sea-lion-v4-27b
48. *LlamaFactory/examples/torch_spark2_5 at main · XHToken/LlamaFactory · GitHub*. https://github.com/XHToken/LlamaFactory/tree/main/examples/torch_spark2_5
49. *Baichuan 2: Open Large-scale Language Models*. https://arxiv.org/html/2309.10305v4
50. *Anytime Pretraining:Horizon-Free Learning-Rate Schedules with Weight Averaging*. https://arxiv.org/html/2602.03702
51. *LlamaFactory/examples/torch_spark2_5/example_1p7b_sft_args.yaml at main · XHToken/LlamaFactory · GitHub*. https://github.com/XHToken/LlamaFactory/blob/main/examples/torch_spark2_5/example_1p7b_sft_args.yaml
52. *LlamaFactory/examples/torch_spark2_5/example_4b_sft_args.yaml at main · XHToken/LlamaFactory · GitHub*. https://github.com/XHToken/LlamaFactory/blob/main/examples/torch_spark2_5/example_4b_sft_args.yaml
53. *LlamaFactory/examples/torch_spark2_5/Spark2_5_Finetune.md at main · XHToken/LlamaFactory · GitHub*. https://github.com/XHToken/LlamaFactory/blob/main/examples/torch_spark2_5/Spark2_5_Finetune.md
