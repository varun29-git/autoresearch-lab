# Engineering plan

Date: 2026-11-12. Status: **superseded by [SOTA_PLAN.md](SOTA_PLAN.md).**

The shard contract in this file still describes the rows. The learning-rate screen does not. Do not run a 250M-token prefix and then continue that cooled checkpoint to 8B. The research plan uses three complete 3B-token cosines, then a new 100B run from the original base. Read SOTA_PLAN.md before writing code.

The original proposal follows, kept so the contract text remains available.

This is how the CPT track should be built if the transition plan is accepted. Data quality, mixture design, licenses, and corpus location are yours. Engineering only defines the tensor contract those files must already satisfy, then trains Spark against that contract.

All of the code below is future work under `cpt_spark_transition/`. Nothing outside this folder is edited, imported, or wrapped. `production/` and `terminus/` stay the from-scratch stack. Their launchers, packers, and checkpoints are the wrong types for this model.

## What we will build

A single-GPU continued-pretraining loop for `XHToken/Spark-X2.6-1.7B-Base`:

- load the pinned Base checkpoint with the stock tokenizer and the vendor remote code
- read an external shard directory
- run AdamW on next-token loss
- checkpoint, kill, and resume
- score a local checkpoint with the letter-logit Indic protocol

The loop is ordinary Python around `transformers`, pinned to 4.57.1 because that is the version named in Spark’s `config.json`. Hugging Face `Trainer` and the vendor LlamaFactory fork are both skipped. Spark’s attention is eager, RoPE ignores `position_ids`, and the tie between `lm_head` and the input embedding has to be visible. A short explicit loop makes those checks local. A framework would hide them.

## What we will not build

- A data mixer, downloader, filter, deduplicator, or decontaminator. The shards arrive finished.
- A tokenizer, a vocab edit, or `resize_token_embeddings`.
- Document packing across documents. One sequence in the shard is one training example. RoPE in `modeling_spark.py` is a monotonic range, so a packed multi-document row would train the full-attention layers at the wrong positions.
- Spectral Muon, FP8, DeepGEMM, or `weight_mode`.
- Multi-GPU. One process, one device. Gradient accumulation is the batch knob.
- SFT, DPO, chat templates, or ternary conversion.
- Any change to files outside this folder. Eval behavior that already exists under `Alternate direction/` is reimplemented here against a local checkpoint path. Those scripts are not modified.

## Shard contract

Training reads one external manifest. The manifest and the arrays stay outside the git checkout. A path inside this repository is refused.

```json
{
  "schema": "spark_cpt_windows_v1",
  "seq_len": 4096,
  "pad_id": 2,
  "bos_id": 0,
  "eos_id": 1,
  "tokenizer_sha256": "<sha256 of the stock tokenizer.json>",
  "shards": [
    {
      "path": "/external/cpt/train-000.int32",
      "rows": 100000,
      "stream": "indic"
    }
  ]
}
```

Each shard is a raw little-endian int32 matrix of shape `[rows, seq_len]`. Row `i` starts at byte `i * seq_len * 4`. Values are Spark token ids in `[0, 131072)`, or nothing else.

A legal row is:

`[bos, content..., eos, pad, pad, ...]`

or, for a window that is not the end of a document:

`[bos, content..., pad, pad, ...]`

`eos` appears only when the row ends a document. `pad` occupies the tail only. `bos` is the first id. Content does not contain ids 0 through 13. Those ids are bos, eos, pad, the think markers, unk, and the chat/FIM markers. The trainer does not repair a row that breaks this. It counts the violation and stops.

`stream` is a label for logging (`indic` or `replay`). The trainer does not rebalance streams. If the manifest is 50:50, that ratio was already written into the rows.

Scored tokens are every non-pad id except the leading `bos`, which the causal shift never predicts. Pad tokens are attention-masked and their labels are −100. The token budget counts scored tokens, not pad slots.

`seq_len` may be lowered to 2048 after the memory smoke. The manifest, the arrays, and the run config then all change together. A checkpoint refuses a resume whose `seq_len` or `tokenizer_sha256` differs.

This format is the whole data surface. Terminus `uint16` shards are not readable here: ids at or above 65,536 do not fit, and end-of-document id 4 is Spark’s `</think>` token.

## Code layout, when implementation starts

```text
cpt_spark_transition/
  spark_cpt/
    contract.py      manifest, row checks, sha256 pins
    batch.py         mask, labels, scored-token count
    schedule.py      warmup / constant / cosine, token-based
    optim.py         AdamW groups, tied weights once
    checkpoint.py    atomic file, contract hash, resume
    load.py          from_pretrained, invariant asserts
    loop.py          one process, accumulate, log jsonl
    eval_mcq.py      letter log-prob, local model path
  tests/
    test_batch.py
    test_schedule.py
    test_checkpoint.py
    test_contract.py
```

Tests use synthetic int32 rows and a tiny `nn.Linear` stand-in. They do not download Spark and they do not allocate the 1.7B model. The real model is exercised only by the smoke command.

## Training step

One microbatch:

1. `input_ids` is the int32 row, cast to long.
2. `attention_mask` is 1 where the id is not pad, else 0.
3. `labels` is a clone of `input_ids`, with pad positions set to −100.
4. Forward with `use_cache=False` and `logits_to_keep` left at the model default only after the smoke confirms which positions the vendor `loss_function` scores. The expected Hugging Face shift is: logit at index `i` predicts `labels[i + 1]`, and index 0 is not a target. The smoke compares that loss to a hand-computed cross-entropy on one known position. If the vendor function already shifts, the loop does not shift again.
5. Backward. Accumulate until the configured scored-token slots are filled (target 64 rows of 4,096, which is 262,144 slots, pads included in the slot count and excluded from the token budget).
6. Clip grads at 1.0. AdamW step. Zero grads.

Parameter groups:

| Group | Params | Weight decay |
| --- | --- | --- |
| Matrices | `q_k_v_proj`, `g_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj` | 0.1 |
| Embeddings | the tied embedding, once | 0 |
| Norms | every `Spark2_5RMSNorm` | 0 |

`lm_head.weight` is not a second Adam state. Load-time assert: `lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()`.

Gradient checkpointing is on. `torch.compile` is off. Precision is bf16 autocast for the step, fp32 Adam moments.

Learning-rate schedule, counted in scored tokens, written for an 8B horizon:

- warmup over the first 80M to the arm’s peak
- constant through 6.4B
- cosine over the last 1.6B down to 0.1× peak

Screen peaks are 1e-5, 3e-5, and 1e-4. Each arm is a separate run directory that stops at 250M scored tokens. The driver does not start the next arm and does not continue the winner. Continuing to 8B is a second command that is given one checkpoint path.

## Checkpoints

External run directory, required on the command line.

A commit writes a temporary file, fsyncs, then renames to `step_NNNNNN.pt` and writes `step_NNNNNN.json` plus `latest.json`. The payload holds model state, Adam state, torch and numpy RNG, scored tokens, microstep, peak LR, and the manifest sha256. `latest.json` moves only forward.

Resume requires the same manifest hash, tokenizer sha256, `seq_len`, peak LR, and model-file hashes recorded at start. A mismatch exits. The smoke deletes the process between two steps and requires the resumed step’s loss to match an uninterrupted process.

Beside the optimizer checkpoint, each screen boundary also exports a weights-only directory (`config`, the pinned modeling files, the stock tokenizer files, bf16 safetensors). Eval and any later ternary or vocabulary project read that export. They do not read the Adam file.

## Eval

`eval_mcq.py` loads a local export. It does not download a fresh Base over a run we just trained. Protocol matches the existing letter-logit harness: few-shot prompt, score the option letters at the last non-pad position, seed 42. Batch size starts at 1 and rises only while memory holds. The first paired run is MILU, ten Indic languages plus English. Other tasks are the same script with a different loader, after MILU runs clean.

Step-0 numbers are measured on this GPU. Older JSON files that contain CUDA out-of-memory strings are not baselines.

## Build order

Each milestone is reviewable on its own. Later milestones wait.

| Milestone | Delivered inside this folder | Done when |
| --- | --- | --- |
| E0 | `contract.py`, `batch.py`, `schedule.py`, `checkpoint.py`, synthetic tests | CPU tests pass. No model download. |
| E1 | `load.py` plus a smoke command | On one training-class GPU: tie holds, one Hindi and one English forward match a second process, pad-vs-unpadded loss on the real tokens matches, peak memory at 512 / 2,048 / 4,096 is written down, `seq_len` is locked. |
| E2 | `optim.py`, `loop.py`, two real steps | Finite loss and grads. Unused special rows stay bitwise equal when weight decay on the embedding is 0. Kill and resume match. |
| E3 | `eval_mcq.py` | MILU on the untouched Base export, batch that fits, every language either a score or a recorded failure. No imputed cells. |
| E4 | Screen command | Three 250M-token runs can be launched by a person, one after another, each in its own external directory. This milestone does not launch them. |
| E5 | Continue command | Given one E4 checkpoint, the same schedule can proceed toward 8B. This milestone does not launch that run. |

E0 is the only milestone that is reasonable to write before a GPU is chosen. E1 decides whether the pilot context is 4,096 or 2,048. E4 and E5 are launch wrappers around the same loop. They are not a scheduler that spends a token budget by itself.

## Commands a person will eventually run

Paths are arguments, never defaults inside the repo.

```bash
python -m spark_cpt.loop smoke \
  --model /external/models/Spark-X2.6-1.7B-Base \
  --seq-len 4096

python -m spark_cpt.loop train \
  --model /external/models/Spark-X2.6-1.7B-Base \
  --manifest /external/cpt/manifest.json \
  --output /external/cpt/runs/screen-3e-5 \
  --peak-lr 3e-5 \
  --stop-scored-tokens 250000000

python -m spark_cpt.eval_mcq \
  --model /external/cpt/runs/screen-3e-5/export \
  --task milu
```

`smoke` and the unit tests are the engineering checks. `train` is a paid run and stays a human action.

## Invariants the code has to keep

These are asserted, not commented.

1. Tokenizer file hash matches the manifest. No save of a tokenizer, no added tokens.
2. `resize_token_embeddings` is never called.
3. Embedding and `lm_head` share storage, and Adam sees that tensor once.
4. Embedding and norm weight decay are 0.
5. A row with an id outside `0 .. 131071`, or with a special id inside the content span, aborts the run.
6. Loss is the mean over labels that are not −100. Two identical contents, one padded, produce the same loss on those labels.
7. Scored-token counter ignores pad and the unpredicted bos.
8. Resume refuses a changed manifest, tokenizer, sequence length, or peak LR.
9. Output directory cannot be inside the git work tree.
10. Modeling-file bytes are hashed at start and stored in the checkpoint. A later ternary or vocabulary project can see which code the weights belong to.

## First implementation slice

When you say to start, the slice is E0 only: the contract, the label mask, the schedule math, the checkpoint round-trip, and CPU tests. No weights, no shards, no GPU, no files outside this folder.
