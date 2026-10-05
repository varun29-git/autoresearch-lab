# How the architecture loop decides

The first campaign trains on **English only**, using the frozen 65K Terminus tokenizer and the same prepared token stream for every candidate. At the 0.5B screen and any matched 2.5B continuation, `status` keeps the following signals separate. No weighted total decides the winner.

| Axis | Measured signal | Interpretation |
| --- | --- | --- |
| Quality | English held-out next-token loss on eight fixed, spaced validation batches; optional answer-only English development loss | Lower at equal tokens is better. Changes within 0.5% are uncertain; >2% regression is flagged. |
| Optimization efficiency | Area under the English validation-loss curve at matched token checkpoints; best and late loss | Lower area means the candidate learned sooner, even if final losses tie. Internal curve evaluations use fewer blocks than the final checkpoint, so this is a supporting signal. |
| Numerical stability | Nonfinite incidents and sampled nonfinite fractions; logged clipping frequency and gradient norms; residual RMS, attention logits/entropy, FP8 input error, update/weight ratios | Nonfinite failure rejects a candidate. Other values are compared with baseline and the proposed mechanism; there is no universal acceptable clipping or entropy threshold. |
| Training cost | Median logged training tokens/s, stage GPU-hours including retries, peak allocated memory, parameter count | A gain that needs much more time or memory is an explicit tradeoff. H100 measurements at 200M are not guaranteed to transfer to 1.5B. |
| Quantization transfer | Optional matched dense-to-ternary QAT probe: immediate conversion loss, short-continuation loss and incidents, with FP8 activations retained | Run only when the design gives a specific reason ternary conversion may change the ranking. The 100M probe cannot prove final ternary quality. |
| Inference cost | Proposal-level estimates of prefill/decode FLOPs, KV-cache bytes per token and expected latency effect | These are reviewed before training. The current runner does not benchmark inference latency; measure it for finalists before choosing a production architecture. |

The fixed-position losses at 0–511, 512–2047 and 2048–4095 tokens are diagnostics for local/global behavior, not a full long-context benchmark. A candidate becomes a finalist when its quality, learning, stability and cost form a defensible tradeoff against the baseline. Indic, code, math, larger-width, and longer QAT transfer are checked on finalists in the next campaign, rather than on every stage-one idea. English is a cost-effective proxy for general architecture research, but transfer to those domains remains to be measured.
