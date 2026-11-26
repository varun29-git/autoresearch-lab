# Spark continued-pretraining transition

Status: **plan for review**. This folder is the only new artifact. No training run, corpus, checkpoint, or ternary conversion is included. The vocabulary cut lives in [vocab/](vocab/).

Read [VOCAB_PLAN.md](VOCAB_PLAN.md) for the device vocabulary cut. Read [SOTA_PLAN.md](SOTA_PLAN.md) for continued pretraining after that checkpoint exists. [TRANSITION_PLAN.md](TRANSITION_PLAN.md) is the earlier decision record. [ENGINEERING.md](ENGINEERING.md) is superseded on the schedule and kept for the shard-row contract. Research writeups are in [research/](research/).

Data preparation stays outside this folder. The trainer consumes finished int32 shards and does not rebalance them.

What this plan recommends, in one paragraph: keep the from-scratch Terminus track untouched. On Spark, delete only scripts outside English and the seven target languages, keep every existing row for Hindi, Marathi, Kannada, Telugu, Tamil, Gujarati, and Bengali, and append the Sarvam pieces those rows do not already cover. Continued pretraining then follows the CPT plan from that checkpoint. Ternary conversion stays later. Embeddings stay tied.

This document does not authorize a cloud VM, an India AI job, or a training sweep. The human launches and stops paid GPU work.
