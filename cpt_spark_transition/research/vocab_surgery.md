# Vocabulary Surgery Without Sacrificing Spark's Protected Behavior

## Executive Summary

- **Exact Loss Is The Binding Constraint**: Copying inherited rows preserves old hidden states and old logits, but adding active output rows changes softmax normalization. Convex-hull initialization can preserve greedy predictions, not the original probabilities or loss. An ordinary resized softmax therefore cannot generally meet the requested exact step-0 contract. [13][6]
- **Frequency Pruning Is Not Structurally Safe**: A rarely emitted intermediate piece may construct a frequent English token. Preserve the byte alphabet, special tokens, protected output tokens, and their merge dependencies - not just frequently emitted pieces. [11][3]
- **Published Precedents Have Different Guarantees**: Swallow extends rather than prunes its vocabulary. Fanar reports Gemma vocabulary pruning but does not disclose an ancestor-safe algorithm or prove English tokenization identity. Neither establishes the requested guarantee. [30][22]
- **Constituent Mean Is A Baseline, Not A Preservation Proof**: Gnani reports HellaSwag changing from **79.0% to 71.5% (published)** immediately after constituent-mean initialization. Its anchored warmup recovers that benchmark, not an exact original distribution. [12]
- **Freezing Must Survive The Optimizer**: Gnani documents inherited-row drift under AdamW despite gradient masking, and restores inherited rows after optimizer steps. This is directly relevant to a partially trainable tied matrix. [12]
- **Adaptation Budgets Are Not Step-0 Guarantees**: Dagan et al. find conventional tokenizer transfer competitive after **50B tokens (published)**. ZeTT substantially reduces adaptation requirements, but its English result after approximately **800M tokens (published)** still trails its original model. [27][16]
- **Route Rather Than Stack Merge Tables**: For the protected-tokenization requirement, an unchanged Spark path for English/code/math is safer than combining unrelated merge priorities. Script-conditional tokenization has primary-source precedents, but none establishes exact preservation for this Spark/Sarvam combination. [17][21][41]
- **Recommended Decision**: Use dependency-safe tokenizer pruning, explicit donor-to-shared-ID translation, exact inherited-row copying, and new-row alignment before CPT. If exact protected loss is non-negotiable, retain the original output support and an explicit original-support execution mode; otherwise acknowledge a relaxed requirement before physically pruning and extending the active softmax.

**Number conventions.** **P** means published in the cited source. **D** means arithmetic derived from published values or the supplied model dimensions. **E** means a proposed/extrapolated Spark acceptance criterion, not a published result. Literature-table numbers are P unless marked D. Spark's vocabulary **131072**, hidden size **2048**, tied embeddings, tokenizer behavior, and control-token requirements are treated as supplied specifications. No literature LR, vocabulary increment, or training budget below is a validated Spark prescription.

## 1. Separate Tokenization, Logits, And Exact Loss

The requested surgery has three distinct invariants:

| Invariant | What is necessary | What it does not establish |
|---|---|---|
| Same tokenization | Same preprocessing, boundaries, applied merges, and token-byte sequence; map IDs if compacting | Same model probabilities |
| Same hidden states and inherited logits | Same input rows, backbone, positions, masks, and numerical execution | Same softmax normalization |
| Same probabilities and loss | Same inherited logits **and original partition function** | Guaranteed future retention after CPT |

The distinction is decisive. Hewitt explicitly gives the following equations for adding one output embedding. For an original token, with original partition function `Z`:

\[
p_{\theta'}(w_i\mid w_{1:i-1})=
\frac{\exp(h^T e_{w_i})}{Z+\exp(h^T e_{n+1})},
\]

and

\[
\operatorname{KL}(p_\theta\|p_{\theta'})=
\log\left(1+\frac{\exp(h^T e_{n+1})}{Z}\right).
\]

These are **published equations**, not an initialization invented for Spark. The added denominator term changes probabilities even when the new token never occurs in the input or target. [13]

For initialization at the **global vocabulary mean**, Hewitt uses

\[
\mu=\frac1n\sum_{i=1}^{n}e_i,\qquad e_{n+1}=\mu.
\]

Jensen's inequality bounds the added contribution relative to `Z` by `1/n`; it makes the disturbance small, not zero. This global-mean result is not automatically a bound of the same strength for a mean over only a token's constituent pieces. [13]

Pruning has the converse problem: an output token need not occur as a protected target to contribute probability mass to the denominator. Removing its row can therefore change protected loss. Replacing removed rows with the same number of new rows does not establish cancellation. These are direct consequences of the published softmax expression, not claims of a special pruning theorem.

The empirical-comparison paper makes the important weaker guarantee explicit: a new embedding in the convex hull of old embeddings cannot exceed the maximum old logit in any direction. This supports preservation of the old greedy choice, with appropriate tie handling; the paper separately acknowledges that normalization and the output distribution change. [6]

**Decision:** There is no source-backed initializer that generally makes physical output-row deletion plus active output-row addition exactly loss-neutral. Under the strict contract, preserve the original output support, not merely its frequently used rows. An explicit protected mode must compute the original-support softmax in original order, exclude appended rows, and retain any otherwise-pruned old rows. That mode is an operational concession, not proof that the ordinary unified expanded model is identical.

## 2. Prune Merge Dependencies, Not Just Emitted Tokens

### What the pruning precedents actually establish

| Primary source | Published operation | Relevance to protected-corpus identity |
|---|---|---|
| **Swallow** | Adds **11,176** Japanese pieces to reach **43,176** tokens; initializes added embedding/output vectors from original-piece averages; introduces NFKC preprocessing | An extension precedent, **not a pruning algorithm**. Its normalization change is unsuitable for an unchanged protected path. [30] |
| **Fanar Prime / Gemma-2-9B** | Describes pruning an original **250,000**-token vocabulary to **128,256** | Reports strategic pruning, but not token-selection rules, ancestor closure, or an English sequence-identity proof. [22] |
| **BPE vocabulary trimming in NMT** | Runs original BPE, then recursively decomposes removable emitted tokens; retains the atomic character set and original merge list | Separates internal merge machinery from the final emitted vocabulary. It can preserve protected sequences if none of their final tokens is decomposed. [3] |
| **Teaching Old Tokenizers New Words** | Prunes leaves in the merge-dependency graph and updates downstream counts | Avoids deleting ancestors of retained descendants. Same BPE inference algorithm does **not** mean every input retains its old segmentation. [11] |

The concrete failure is **Kentuc -> Kentucky**: the trimming paper notes that `Kentuc` can disappear from final tokenizations after `Kentucky` is learned, while remaining necessary to construct it. Deleting its creating merge merely because its emitted frequency is low makes the retained English descendant unreachable. [3] The newer leaf-pruning paper independently explains why frequency pruning breaks merge paths and creates unreachable tokens. [11]

### Proposed protected-corpus procedure

1. **Freeze the original tokenizer implementation and configuration.** Preserve its byte mapping, regex, normalization, added-token matching, prefix-space behavior, and merge ordering. GPT-2's reference code explicitly separates regex matching, UTF-8 byte mapping, ranked BPE, and ID lookup. [35]
2. **Build the protected keep set** from English, code, and math tokenizations. Retain all emitted pieces, all symbols and merge rules used in their original merge traces, and recursively required ancestors. Preserve original relative merge ranks.
3. **Always retain the complete byte alphabet and every control token.** Do not frequency-prune bos/eos/pad, chat, think, or FIM markers. Their semantic importance is not measured by ordinary corpus frequency. The GPT-2 reference byte mapping supplies byte coverage independently of learned merges. [35]
4. **Prune only outside that closure.** Prefer dependency-safe leaf deletion. Alternatively, retain original BPE internally and post-decompose only selected final tokens, as in the NMT trimming approach. Do not rebuild a tokenizer with dangling merge references.
5. **Make ID policy explicit.** Keeping original IDs requires retaining slots. Physical compaction requires a bijective old-to-new map for survivors, remapped embedding rows, special-token configuration, datasets, templates, and checkpoint metadata. Donor IDs are never this map.

For acceptance, require **100% exact sequence agreement (E)** on the protected corpus: compare token-byte strings, offsets, and original IDs after reversing any compaction map. Report raw-ID agreement separately; compaction intentionally changes many numeric IDs.

Fertility and bytes-per-token are useful compression diagnostics, but equal averages cannot prove sequence identity. Bits-per-byte should be reported as total teacher-forced negative log2 probability divided by original UTF-8 bytes; it tests likelihood, not merge identity. The paper's Self-Tokenization Test detects unreachable tokens, not protected-corpus equivalence. [11][27]

**Decision:** Protect merge traces first, then optimize vocabulary size. A finite protected suite establishes identity on that suite, not on every unseen English/code/math string.

## 3. Initialization Equations: What Is Actually Published

Let `E_s` denote original embeddings. The equations below reproduce the cited methods using compact notation; their original settings are often tokenizer **replacement**, not append-only extension.

### FOCUS: overlap copying plus sparse similarity weights

FOCUS copies overlapping rows and initializes additional rows from overlapping source rows:

\[
e_o^t=e_o^s,\quad
c_a=[\cos(x_a,x_{o_1}),\ldots,\cos(x_a,x_{o_n})],
\]
\[
w_a=\operatorname{sparsemax}(c_a),\qquad
 e_a^t=\sum_{o\in O}w_{a,o}e_o^s.
\]

The auxiliary `x` vectors are fastText embeddings trained on target-tokenized text. The transferred vectors are the pretrained model embeddings, not the fastText vectors. [9]

### WECHSEL: cross-lingual nearest neighbors

Using aligned static subword vectors, WECHSEL publishes

\[
s_{x,y}=\frac{u_t(x)u_s(y)^T}{\|u_t(x)\|\|u_s(y)\|},
\]
\[
e_t(x)=
\frac{\sum_{y\in J_x}\exp(s_{x,y}/\tau)e_s(y)}
{\sum_{y'\in J_x}\exp(s_{x,y'}/\tau)}.
\]

Its reported choices are **k = 10, tau = 0.1 (P)**. It replaces the English vocabulary for target-language specialization; it does not promise to preserve inherited rows simply because their spellings overlap. [15]

### OFA: factorized coordinates, then similarity transfer

OFA uses

\[
E_s\approx F_sP,
\]

copies shared coordinates, and initializes other coordinates by

\[
F_t[y]=
\frac{\sum_{x\in N(y)}\exp(s(x,y)/\tau)F_s[x]}
{\sum_{x'\in N(y)}\exp(s(x',y)/\tau)}.
\]

The reported defaults are **k = 10, tau = 0.1 (P)**. The source factorization is approximate. Installing that factorization is not the requested unchanged Spark architecture, and reconstructing old rows through it is not exact row preservation. An offline new-row-only variant would be an extrapolation. [2]

### FVT and the ordinary constituent mean

FVT explicitly copies shared tokens and, for a new token, publishes

\[
E_{in}(t_i)=\frac{1}{|\mathcal T_{gen}(t_i)|}
\sum_{t_j\in\mathcal T_{gen}(t_i)}E_{gen}(t_j).
\]

This is the concrete primary-paper basis for the old-tokenizer-piece mean. It uses the actual original tokenization, not all conceivable decompositions. [8]

**VIP naming caution:** The primary method I could substantiate is **VIPI**, Vocabulary Initialization with Partial Inheritance. Its algorithm selects partitions with the fewest pieces, then those containing the longest piece, averages within eligible partitions, and averages across partitions. The paper gives pseudocode, **not a standalone initialization equation**. I could not uniquely substantiate a distinct vocabulary initializer called `VIP`; it should not be silently conflated with unrelated VIP prompting papers. [26]

**Multi-VT uncertainty:** I could not uniquely identify a primary method under that exact name. FVT, VIPI, multilingual transfer, and multivariate Gaussian initialization are not interchangeable labels. No equation, alignment budget, or English-retention result is assigned to `Multi-VT` here.

### ZeTT: learned tokenizer-to-embedding transfer

ZeTT publishes the mapping and training objective

\[
H_\theta:(V_b,T_b)\longrightarrow\phi_b,
\]
\[
L_{final}=L_\theta(T_b(x),H_\theta(V_b,T_b),\psi)
+\alpha L_{aux}.
\]

The non-embedding LM parameters `psi` remain fixed during hypernetwork training. The overlap auxiliary loss encourages predicted shared rows toward their originals; it is **not exact copying**. [16]

**Decision:** For Spark, copy inherited rows exactly irrespective of the chosen initializer. Apply a learned or heuristic method only to genuinely new rows. This is a deliberate append-only adaptation of the methods, not a guarantee supplied by their replacement experiments.

## 4. Published Recipes And Their Actual Preservation Scope

In the tables, **H** means hidden states and inherited raw logits can remain identical on unchanged tokenizations **if** inherited rows/backbone are copied exactly. It does **not** mean identical probabilities or loss. **NR** means not reported or not established in that source, never zero. LR entries identify the stage to which they belong.

### General transfer methods

| Method | Vocabulary change | Embedding-only budget | LR | Backbone frozen? | Step-0 forward preservation | Reported English effect |
|---|---|---|---|---|---|---|
| **FOCUS** | Target replacement vocabulary **50k**; new-row count depends on overlap | No separate FOCUS alignment phase; total LAPT **12.8B tokens** | **5e-5**, LAPT | Not a new-row-only frozen-backbone recipe | H for copied overlap-only inputs; full probabilities no | English evaluation NR. [9] |
| **WECHSEL** | Replacement vocabulary **50k** | No embedding-only phase for WECHSEL; **250k** total training steps | RoBERTa **1e-4**; GPT-2 **5e-4**; initial **10%** LR warmup | Full-model training; embedding-only freezing belongs to its TransInner baseline | Not generally H: rows are semantically reinitialized; probabilities no | Source-language preservation is not the objective; matched drop NR. [15] |
| **VIP / substantiated VIPI** | Replacement experiments include **8k, 16k, 32k** vocabularies | **One epoch MLM** before task training; token count NR | Alignment LR NR | Frozen-backbone schedule NR | H conditional on copied overlap; probabilities no | General English capability drop NR. [26] |
| **OFA** | Target vocabulary **401k** | No separate embedding-only budget; early-stopped MLM | **5e-5** initial CPT LR | No comparable new-row-only freeze reported | Not exact with approximate factorization; probabilities no | Matched English before/after NR. [2] |
| **ZeTT** | Target-dependent replacement vocabulary | Hypernetwork **200k steps**, including **10k** imitation warmup; target adaptation **819.2M tokens** | Main hypernetwork peak **6e-5**; imitation warmup peak **3e-4** | Backbone fixed during hypernetwork training; target adaptation trains full model | Shared rows predicted/regularized, not exactly copied; probabilities no | English average **76.9 original -> 73.0 zero-shot -> 74.9 after ~800M**. [16] |
| **FVT** | Replacement/compression; no universal appended count | **One epoch MLM** | **3e-5** reported for downstream fine-tuning, not a demonstrated alignment LR | No new-row-only freeze recipe | H conditional on copied overlap; probabilities no | CoNLL03 F1 changes are task-specific, not broad English-LM retention. [8] |
| **Multi-VT** | Exact primary identifier unresolved | NR | NR | NR | Unestablished | Unestablished |
| **Heuristic old-piece mean** | Any selected appended set; no standard count | No inherent training recipe | Must be chosen for Spark, E | Can freeze backbone and inherited rows, E | H before updates; probabilities no | No universal drop; Gnani provides a counterexample below. [8][12] |

The table exposes a recurring category error: an LR warmup is not an embedding-only alignment stage, and a replacement-vocabulary experiment is not evidence of append-only English preservation.

### Language-adaptation case studies

| Project | Added tokens and initialization | Alignment-only amount | LR | Frozen parameters | English effect and forward-pass scope |
|---|---|---|---|---|---|
| **Nanda** | **128,256 -> 153,856** total; **25,600 added, D**. Weighted average of **top-5** original tokens retrieved with text-embedding-3-large. **Paper does not state the weighting equation.** | Separate vocabulary-alignment budget NR | Full recipe peak **1.5e-4** | Original backbone frozen, but additional decoder blocks introduced | **Not an unchanged architecture**. English fertility **1.35 -> 1.35** is not loss identity; matched English before/after drop NR. [4] |
| **DictaLM 2.0** | **1,000** Hebrew tokens. **No explicit new-row initialization equation.** | Approximately **500k sentences**, one epoch distillation; then **100k documents**, one epoch head calibration | **5e-6** in both phases | First phase: only new input rows; second: embedding and head layers train | H conditional at surgery, not full-probability preservation; calibration can change inherited layers. English stage-wise drop NR. [10] |
| **Krikri** | **20,992** added; **149,248** total. **No explicit initialization equation.** | **5B tokens** | Embedding-stage LR NR | Only new embedding/output rows train initially; then full unfreeze | H conditional before full CPT; probabilities no. English average **66.2 -> 62.2** after initial CPT; curated/synthetic additions yield final **67.0**. [28] |
| **Gnani Evon** | **32,000** added: **131,072 -> 163,072**. Explicit constituent means, separately for untied input and output rows | Approximately **3B tokens**, **90% Indic / 10% English** | Alignment peak **1e-4** | Backbone and inherited rows frozen; optimizer-step restoration used | H conditional, probabilities no. Immediate HellaSwag **79.0 -> 71.5**; anchored warmup **79.0**. Final CPT nine-benchmark mean **70.62 -> 67.95**. [12] |
| **Chinese-LLaMA** | Donor vocabulary **20,000**; merged total **49,953** from **32,000**, hence **17,953 net added, D**. **No explicit new-row initialization equation.** | Basic 7B stage trains embeddings with transformer fixed; token count NR | Listed two-stage peaks **2e-4 / 1e-4**; source text does not explicitly assign the pair | Embedding stage followed by embeddings/head/LoRA training | Original rows appended/preserved at surgery, so H conditional; later whole-embedding training need not preserve them. English drop NR. [23] |
| **Tamil-LLaMA** | **16,000** added; **48,000** total. **No explicit initialization equation.** | No separately quantified embedding-only stage | Pretraining **2e-4** | Embeddings, head, and LoRA parameters train; no inherited-row freeze specified | H conditional at surgery; probabilities no. English drop NR. [14] |
| **OpenHathi** | **16k** added: **32k -> 48k**; random new rows. **No explicit initialization equation.** | Translation alignment then bilingual next-token training; embedding-only token count NR | NR in the primary blog | Low-rank adapters used; no precise new-row-only freeze recipe | H conditional at surgery; probabilities no. Reports a small English degradation qualitatively, not an exact numerical drop. [20] |

**DictaLM illustrates why alignment is useful but not directly portable.** It explicitly publishes the representation-distillation objective

\[
L=\|h_{old}-h_{new}\|_2,
\]

using final hidden states for the same sentence under old and extended tokenizations. Its later head-calibration phase is separate because new output rows remain random after input-side tuning. Spark's tied matrix cannot reproduce that independent input/head sequence literally. A tied adaptation must optimize the single shared matrix and acknowledge the changed objective. [10]

**Gnani is the closest operational warning.** It publishes constituent-mean formulas for both matrices, preserves inherited rows, and still observes an immediate English benchmark drop. A freeze-leak run falls to **34.0 HellaSwag (P)**; the anchored run recovers to **79.0 (P)**. The mechanism is not just semantic alignment: new output rows compete with old rows, and optimizer drift can silently move supposedly frozen parameters. [12]

**Nanda is a useful contrast, not a Spark recipe.** Its retrieved weighted-average initialization is relevant, but its added-block training changes model architecture. Its published English comparisons are also not a matched surgery experiment: for example, Nanda-10B's English average is **57.93 (P)** versus **56.83 (P)** for Llama-3-8B-Instruct. Different model sizes and training histories preclude treating that comparison as a step-0 drop estimate. [4]

**Decision:** The strongest transferable practice is exact inherited-row copying plus explicitly audited freezing. None of these projects demonstrates the complete Spark requirement: tied embeddings, unchanged protected tokenization, unchanged full softmax, and identical step-0 protected loss.

## 5. Dagan And ZeTT Do Not Supply A Neutrality Deadline

Dagan et al. evaluate tokenizer changes on a **1.5B model (P)** over budgets including **5B through 500B tokens (P)**. Changed tokenizers become competitive with an unchanged-tokenizer model after **50B tokens (P)** in their code-generation setup. Importantly, their main transfer uses **FVT**, not a pure random initializer. [27]

Their initialization ablation reports HumanEval Pass@1 of **20.5% with FVT**, **18.4% without FVT**, and **20.8% for the merged-tokenizer condition (all P)**. The corresponding MBPP values are **27.2%, 25.0%, and 27.6% (P)**. These are downstream results, not equality of original English loss, and the cited ablation does not justify relabeling the unspecified without-FVT initializer as a universally representative random baseline. [27]

ZeTT changes the economics by learning a reusable tokenizer-to-embedding mapping. Its target adaptation runs **50,000 steps**, batch **32**, sequence length **512**, or **819.2M tokens (all P)**. But the hypernetwork has already been trained for **200,000 steps (P)**; zero-shot transfer is not zero prior training cost. [16]

| Published comparison | Original | Immediately after transfer | After approximately 800M adaptation tokens |
|---|---:|---:|---:|
| Mistral-7B English average, FOCUS initialization | 76.9 | 55.6 | 68.7 |
| Mistral-7B English average, ZeTT | 76.9 | 73.0 | 74.9 |
| Code Pass@1 average, ZeTT | 28.1 | 23.2 | 27.6 |

All table values are P. English transfer uses the GPT-2 tokenizer; code transfer uses StarCoder. They are separate evaluations, not a common-tokenizer preservation experiment. [16]

The learned method substantially narrows the adaptation gap. It does not reach exact original English benchmark performance in the shown checkpoint, much less prove token-wise loss equality. The literature therefore supports **tens of billions of tokens for conventional transfer in one setup versus less than a billion target-adaptation tokens for learned transfer in another**, not a universal random-versus-learned break-even rule. [27][16]

**Decision:** If step-0 loss equality is mandatory, no later token budget repairs a failure of that initial contract. Treat published budgets as planning precedents for recovery experiments, never as acceptance deadlines or guaranteed Spark alignment counts.

## 6. Use An Indic Router With Explicit Boundary Semantics

For this deployment, the allowed native-script paths are Devanagari for **Hindi and Marathi**, and the respective **Kannada, Telugu, Tamil, Gujarati, and Bengali** scripts. English and Latin-script code-mix stay on Spark's original path. Code and math need an explicit protected content mode, because their string literals, identifiers, comments, or notation can contain Indic characters.

| Hazard | Why it matters | Proposed handling, E |
|---|---|---|
| **Dependent vowel sign at a token prefix** | Unicode stores consonants and dependent vowels in logical order even when rendering reorders them. A token beginning with a mark can be a legitimate continuation. [33] | Do not blanket-delete mark-prefixed pieces. Keep routing boundaries outside orthographic clusters; test continuation use separately from string-initial use. |
| **NFC versus NFKC** | NFC preserves canonical equivalence; NFKC also performs compatibility decomposition and may erase meaningful distinctions. [36] | Match donor NFC only on the donor path. Do not introduce NFKC, case folding, or whitespace normalization on protected Spark input. |
| **Nukta** | Some nukta letters, including U+0958, are composition exclusions. NFC does not mean every decomposed spelling becomes one precomposed character. [36] | Test canonical equivalents, decomposed forms, and mark ordering. Do not equate visual equality with byte equality. |
| **ZWJ/ZWNJ and virama** | Joiners affect half forms and conjunct formation; Marathi has language-specific RA behavior. [33] | Preserve joiners; never strip them as generic cleanup. Attach them to the surrounding Indic routing span and test relevant clusters. |
| **Latin-script code-mix** | Script cannot distinguish English from Romanized Hindi or Marathi | Preserve Latin spans on Spark unless a separately approved language-aware policy changes this contract. |
| **Leading U+0120 and whitespace** | GPT-style byte mapping and pretokenization determine whether a space belongs to a token. [35] | Translate donor token representations through their actual byte codec; do not treat the visible marker as ordinary text or strip it. Preserve leading-space/no-space variants. |
| **Global merge-table concatenation** | New intermediate merges can preempt old paths; unrelated score scales have no common semantic ordering | Keep separate merge engines behind a deterministic router. Do not assume donor merge scores are comparable with Spark ranks. |

The router should return IDs in **one shared Spark vocabulary**, through explicit donor-ID-to-shared-ID translation. Deduplicate by exact token bytes and boundary semantics, not donor integer IDs or lossy Unicode decoding. Preserve all donor-internal dependencies needed by imported pieces, even if an intermediate symbol is never emitted.

There are relevant primary precedents, but their scope matters. **SCRIPT** conditions pretokenization on Unicode script and character category, grouping Letters and Marks and attaching Inherited-script groups; it is not a drop-in two-BPE router. **AdaptTok-Indic** explicitly routes among Bengali, Devanagari, and shared boundary predictors, but uses a different representation and does not prove preservation of an existing English tokenizer. [17][21]

**MUTANT provides counterevidence to naive merging:** its script-specific vocabulary rule-stacking ablation introduces distributional interference and inconsistent segmentation. Its successful jointly trained tokenizer is a useful training-from-scratch result, not an exact-preservation solution for an existing Spark checkpoint. [41]

**Decision:** A router is safer **for the protected segmentation contract**, not inherently better for every language-model metric. Entire protected documents must bypass routing unchanged. Mixed-content routing requires explicit whitespace, punctuation, numeral, joiner, and code-scope tests. NFC-normalized Indic input cannot also be promised to round-trip to its original non-NFC bytes without an additional preservation mechanism.

## 7. Resize The Tied Matrix, Align, Then Start CPT

### Surgical checkpoint transformation

1. **Create a reversible manifest.** Record original tokenizer artifacts, special IDs, checkpoint hashes, kept/pruned old IDs, and donor-to-shared-ID mappings. Exclude donor control tokens and non-target-language additions. The imported vocabulary size must be selected from measured per-language coverage and compression, not copied from an unrelated project.
2. **Choose the exactness contract before allocating rows.** For strict protected-loss identity, keep every original output row and provide an original-support execution mode. For physical compaction plus a fully active expanded softmax, explicitly accept that exact original loss is not guaranteed.
3. **Allocate one new tied matrix.** Copy each surviving old row into its mapped location, bit-for-bit. Leave architecture, hidden dimension, RoPE, attention, and backbone tensors unchanged. The original matrix contains **268,435,456 elements (D)**; each **1,000** appended rows adds **2,048,000 elements (D)** from the supplied dimensions.
4. **Initialize only new rows.** Without a compatible learned hypernetwork, use the FVT constituent mean as the transparent baseline. Obtain pieces with the frozen original tokenizer under the correct leading-space context. For byte pieces that are not independently valid UTF-8, compose through the byte-level representation rather than decoding with replacement characters. [8][35]
5. **Restore and verify the tie.** Input lookup and output projection must reference the same parameter/storage, not merely equal-valued copies. Check again after saving and reloading. Transformers documents resizing and retying, but ordinary resizing adds/removes rows **at the end**; it is not arbitrary ID compaction. [29]

Do not confuse constituent averaging with `mean_resizing=True`: Transformers documents the latter as sampling from a multivariate normal with the old matrix's mean and covariance. It is not exact per-token mean pooling. [29]

### Is the constituent mean the right default?

It is a defensible **baseline**, not an established optimum. FVT provides the direct formula; Swallow and Gnani supply decoder adaptation precedents. The empirical comparison also finds simple mean/multivariate schemes competitive with more complex initializers. [8][30][12][6]

Its limitations are concrete. Equal weight per piece is not equal weight per character: a short fragment receives the same coefficient as a long fragment. Averaging can reduce vector norms and change output-logit scale. A compound token's output row is not thereby calibrated to the probability of producing the original multi-token sequence. Global-mean probability bounds and constituent-mean semantic heuristics must not be conflated. [5][13]

The 2026 systematic study finds a better configuration using uniform input composition, input norm calibration, and character-length-weighted output composition. It reports validation loss **2.722 versus 2.806 after 50 CPT steps (P)** for that configuration versus symmetric uniform composition. However, the experiment uses **untied** embeddings; the asymmetric solution cannot be installed unchanged in Spark. [5]

### Adam state and reliable freezing

**Recommended default, E:** construct a fresh optimizer after surgery, with an explicit alignment-stage schedule. Do not load old embedding state tensors by position into a differently shaped parameter. PyTorch Adam maintains tensor-shaped first/second moments, optional maximum second moments, and a scalar step per parameter. [31]

If preserving optimizer history, remap old moments with exactly the row map, initialize new-row moments to zero, and replace optimizer parameter references. Preserve unchanged backbone state only deliberately. A standard Adam scalar step cannot simultaneously give inherited and new rows independent optimizer ages; zero moments plus an old scalar step is not the same as freshly initialized row-wise Adam. [31]

During alignment, freeze the backbone and inherited rows of the **single shared matrix**. Mask gradients from both input and output uses after accumulation. A zero row gradient is insufficient protection against existing momentum or decoupled weight decay: use a row-aware update or snapshot/restore inherited rows, including master/sharded copies, and audit moment state. This is supported by both Adam's update code and Gnani's observed freeze leak. [31][12]

### Alignment precedents, not a Spark token prescription

| Published project | Vocabulary-alignment amount | What it establishes |
|---|---|---|
| DictaLM | Approximately **500k sentences**, then **100k documents**, each one epoch | Representation alignment followed by separate head calibration; not a token-count budget. [10] |
| Gnani | Approximately **3B tokens** | New-row-only anchored warmup on an untied model. [12] |
| Krikri | **5B tokens** | New embedding/output rows train before full-model CPT. [28] |
| ZeTT | **200k hypernetwork steps**; **819.2M target-adaptation tokens** | Reusable learned initialization followed by full-model adaptation, not a new-row-only alignment recipe. [16] |

These precedents do not establish a universally "short" alignment stage for a tied 1.7B model. Use an English/code/math anchor plus exactly the seven target Indic languages, evaluate the frozen-row alignment trajectory, and select its length empirically. Do not import Nanda's added blocks, Gnani's long-context changes, or an untied output initializer.

**Decision before the first full-model update:** tokenizer and ID mappings must be final; the saved/reloaded tie must be correct; inherited tensors must survive an optimizer dry run unchanged; new rows must have finite, usable representations; all acceptance gates below must pass under the chosen contract. Only then unfreeze for full CPT. Alignment cannot retroactively make a failed step-0 test pass.

## 8. Synthesis: The Recommended Procedure And Acceptance Gates

| Strategy | Mechanism | Best scope | Principal trade-off | Evidence and time horizon |
|---|---|---|---|---|
| Dependency-safe pruning | Removes only dispensable emitted pieces/merge leaves outside protected closure | Protected tokenizer compression | Physical output pruning still changes loss | Structural evidence at surgery; quality still needs testing. [11] |
| Constituent mean plus inherited-row freeze | Reuses old representations and aligns only new rows | Transparent append-only baseline | Output competition remains; tied input/output needs compromise | FVT formula; Gnani shows both recovery and failure modes. [8][12] |
| Learned transfer | Predicts semantically/contextually useful replacement rows | Faster adaptation across tokenizers | Training cost; approximate overlap preservation; not exact step-0 neutrality | ZeTT improves zero-shot and sub-billion-token adaptation. [16] |
| Script router plus protected support mode | Preserves original tokenization path and original output denominator when selected | Strict English/code/math compatibility | Custom tokenizer/runtime mode; retained old output storage | Script-conditional precedents support design, not this full combined guarantee. [17][21] |

The non-obvious tension is that the best memory-saving operation and the strongest preservation contract point in opposite directions. Tokenizer pruning can remove emitted symbols without removing their internal merge machinery; exact probability preservation can likewise require retaining output rows that are never protected targets. "Unused" has different meanings in tokenization, input lookup, and softmax normalization.

**Recommended procedure, E:**

1. Keep an immutable untouched Spark baseline.
2. Build the protected merge closure and prune only outside it. Under the strict loss contract, make this tokenizer-level pruning rather than deleting original output support.
3. Import only target Indic pieces through a script/content router, with explicit byte-level and ID translation. Leave English, Latin code-mix, code, math, and control-token processing untouched.
4. Copy inherited rows exactly; initialize new tied rows by original-piece mean as the baseline, or a compatible learned new-row-only initializer.
5. Save/reload and run surgery-time acceptance tests **before training**.
6. Align only new rows with the backbone and inherited rows effectively frozen, including optimizer effects.
7. Re-run the gates, then begin full CPT with retention evaluation. If a single ordinary compacted/expanded softmax is required, replace "identical loss" with an explicitly agreed measurable tolerance; do not call that exact preservation.

### Acceptance tests

All zero/100% thresholds below are **E**, chosen to express the requested exactness contract.

- **Vocabulary/control manifest:** all required bos/eos/pad/chat/think/FIM markers retain identities and behavior; no donor special IDs leak into the shared ID space.
- **Exact protected tokenization:** **100%** agreement in token bytes, boundaries, and canonical original IDs across English, code, and math; include whitespace, leading spaces, escapes, indentation, FIM, chat/think text, and literal special-token spellings.
- **Reachability:** no dangling merge references; **zero** unreachable retained ordinary tokens under the appropriate self-tokenization test. Verify the byte alphabet independently of ordinary UTF-8 text round trips. [11]
- **Unicode/donor correctness:** test every target language, nukta variants, vowel signs, virama clusters, ZWJ/ZWNJ, leading-space variants, mixed scripts, Latin transliteration, and Indic text inside protected code. Distinguish raw-byte round trips from donor-NFC round trips.
- **Tensor identity:** inherited rows and all backbone tensors are bitwise unchanged at surgery. Verify the tie by shared storage and again after checkpoint reload.
- **Hidden-state/raw-logit identity:** compare every protected position with the untouched base, after reversing any ID map. Use identical positions, masks, dtype, evaluation mode, and deterministic execution. A changed matrix shape can change numerical kernels, so mathematical equality alone does not certify bitwise implementation equality.
- **Probability-support identity:** compare the complete original-support distribution, not just top-1 output. The protected execution must retain all original denominator terms and assign **zero active probability** to appended rows. Masking only new rows is insufficient if old rows were physically deleted.
- **Exact loss:** teacher-forced token-wise negative log-likelihood and corpus bits-per-byte must match the untouched model under the same original support and numerical reduction path. Fertility equality or an unchanged benchmark score cannot substitute for this test.
- **Optimizer dry run:** verify inherited-row and backbone hashes after a real alignment update with the actual optimizer, weight decay, mixed-precision master weights, distributed sharding, and checkpoint restart.
- **Retention after alignment/CPT:** evaluate English, code generation, and arithmetic separately. Numerical segmentation is a known source of systematic arithmetic differences, so unchanged text preprocessing is not an adequate proxy. [42]

**Uncertainties:** There is no published Spark-specific optimum for added-token count, pruning depth, alignment LR, or alignment tokens. The Sarvam/Spark codec and boundary equivalence must be established from their actual artifacts. VIP and Multi-VT remain unresolved exact method identifiers. Several adaptation papers omit initialization equations or alignment token counts, and many use untied matrices, LoRA, architecture expansion, or different languages. Finally, passing a finite corpus suite is not a universal theorem about unseen strings.

**Bottom line:** The requested step-0 equality is attainable only by preserving the original computation and original output support on the protected path. A physically pruned, actively extended ordinary tied softmax is a different distribution. No cited mean initializer, retrieval initializer, distillation method, or later CPT budget removes that distinction.

## References

1. *Getting the most out of your tokenizer for pre-training and domain adaptation*. https://proceedings.mlr.press/v235/dagan24a.html
2. *O : A Framework of Initializing Unseen Subword Embeddings for Efficient Large-scale Multilingual Continued Pretraining*. https://aclanthology.org/2024.findings-naacl.68.pdf
3. *An Analysis of BPE Vocabulary Trimming in Neural Machine Translation*. https://arxiv.org/html/2404.00397
4. *Nanda Family: Open-Weight Generative Large Language Models for Hindi*. https://aclanthology.org/2026.eacl-long.288.pdf
5. *Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary Extension*. https://arxiv.org/html/2608.03494v1
6. *An Empirical Comparison of Vocabulary Expansion and Initialization Approaches for Language Models*. https://arxiv.org/html/2407.05841
7. [
    
      Initializing New Word Embeddings for Pretrained Language Models · John Hewitt
    
  ](https://www.cs.columbia.edu/~johnhew//vocab-expansion.html)
8. *Fast Vocabulary Transfer for Language Model Compression*. https://arxiv.org/html/2402.09977v1
9. *F : Effective Embedding Initialization for Monolingual Specialization of Multilingual Models*. https://aclanthology.org/2023.emnlp-main.829.pdf
10. *Adapting LLMs to Hebrew: Unveiling DictaLM 2.0 with Enhanced Vocabulary and Instruction Capabilities*. https://arxiv.org/html/2407.07080
11. *Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models*. https://aclanthology.org/2026.findings-eacl.341.pdf
12. *Contents*. https://arxiv.org/html/2610.15371v1
13. [
    
      Initializing New Word Embeddings for Pretrained Language Models · John Hewitt
    
  ](https://nlp.stanford.edu/~johnhew/vocab-expansion.html)
14. *Tamil-Llama: A New Tamil Language Model Based on Llama 2*. https://arxiv.org/html/2311.05845v1
15. *M*. https://arxiv.org/pdf/2112.06598
16. *Zero-Shot Tokenizer Transfer*. https://papers.nips.cc/paper_files/paper/2024/file/532ce4fcf853023c4cf2ac38cbc5d002-Paper-Conference.pdf
17. *BPE Stays on SCRIPT: Structured Encoding for Robust Multilingual Pretokenization*. https://arxiv.org/pdf/2505.24689
18. *Multilingual Tokenization through the Lens of Indian Languages: Challenges and Insights*. https://aclanthology.org/2026.findings-acl.1632.pdf
19. *Swallow (Llama 2) — Swallow LLM*. https://swallow-llm.github.io/swallow-llama.en.html
20. *OpenHathi Series*. https://www.sarvam.ai/blogs/openhathi-series
21. *AdaptTok-Indic: Script-Aware Adaptive Tokenization for Indic Languages*. https://arxiv.org/pdf/2610.33017
22. *Fanar An Arabic-Centric Multimodal Generative AI Platform*. https://arxiv.org/pdf/2501.13944
23. *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca*. https://arxiv.org/html/2304.08177
24. *Llama 3 Swallow — Swallow LLM*. https://swallow-llm.github.io/llama3-swallow.en
25. *Efficient Vocabulary Reduction for Small Language Models*. https://aclanthology.org/2025.coling-industry.64.pdf
26. *Fine-Tuning Transformers: Vocabulary Transfer*. https://arxiv.org/pdf/2112.14569
27. *Getting the most out of your tokenizer for pre-training and domain adaptation*. https://raw.githubusercontent.com/mlresearch/v235/main/assets/dagan24a/dagan24a.pdf
28. *Krikri: Advancing Open Large Language Models for Greek*. https://aclanthology.org/2025.findings-emnlp.268.pdf
29. *Models · Hugging Face*. https://huggingface.co/docs/transformers/v4.56.1/en/main_classes/model
30. *Continual Pre-Training for Cross-Lingual LLM Adaptation:Enhancing Japanese Language Capabilities*. https://arxiv.org/html/2404.17790v1
31. *raw.githubusercontent.com*. https://raw.githubusercontent.com/pytorch/pytorch/main/torch/optim/adam.py
32. *Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models*. https://arxiv.org/html/2402.14714v1
33. *Chapter 12 – Unicode 17.0.0*. https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-12
34. *Fast Vocabulary Transfer for Language Model Compression*. https://aclanthology.org/2022.emnlp-industry.41.pdf
35. *gpt-2/src/encoder.py at master · openai/gpt-2 · GitHub*. https://github.com/openai/gpt-2/blob/master/src/encoder.py
36. *UAX #15: Unicode Normalization Forms*. https://www.unicode.org/reports/tr15/tr15-57.html
37. *Vocabulary Expansion of Large Language Models via Kullback-Leibler-Based Self-Distillation*. https://arxiv.org/pdf/2508.15807
38. *Fine-tuning transformers: Vocabulary transfer - ScienceDirect*. https://www.sciencedirect.com/science/article/pii/S0004370223000061
39. *Trans-Tokenization and Cross-lingual Vocabulary Transfers: Language Adaptation of LLMs for Low-Resource NLP A X*. https://arxiv.org/pdf/2408.04303
40. *An Empirical Study on Cross-lingual Vocabulary Adaptation for Efficient Language Model Inference*. https://aclanthology.org/2024.findings-emnlp.396.pdf
41. *MUTANT: A Recipe for Multilingual Tokenizer Design*. https://aclanthology.org/2026.acl-long.2146.pdf
42. *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs*. https://arxiv.org/pdf/2402.14903
