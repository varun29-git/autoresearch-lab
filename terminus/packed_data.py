"""Finite packed curriculum adapter, batching and full CPU acceptance.

The manifest's total_train_tokens denotes input slots; scored_train_tokens
denotes actual labels. Both are explicit in the verified data contract.
Packed candidates live inside provenance['packed_stream']; no source data is
embedded. Each ordered phase retains its original finite mixture permutation.
"""
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from dataclasses import asdict
from pathlib import Path
import threading

import numpy as np
import torch

from production.packed_mixture import PackedMixture, PackedSequenceReader
from .document_attention import DocumentLayout


def stream_spec(manifest):
    spec = manifest.provenance.get("packed_stream", {})
    if not manifest.is_packed or spec.get("schema") != 1:
        raise ValueError("packed training requires a versioned document-segment stream")
    if type(spec.get("tokens_before", 0)) is not int or spec.get("tokens_before", 0) < 0:
        raise ValueError("packed lifetime token offset must be a nonnegative integer")
    if manifest.config.source_mix or manifest.config.phase_source_mixes or manifest.config.prefix_source:
        raise ValueError("packed curricula own their finite hierarchical mixing; continuous routing is incompatible")
    if manifest.config.total_train_tokens != manifest.config.usable_train_tokens:
        raise ValueError("packed input-slot horizon must contain complete sequences")
    for split in ("train", "val"):
        phases = spec.get(split)
        if not isinstance(phases, list) or not phases:
            raise ValueError("packed manifests need train and heldout document phases")
        for phase in phases:
            if (phase.get("schema") != "ci1_packed_mixture_candidate_v1" or
                    phase.get("repeats_sequences") is not False or not phase.get("shards")):
                raise ValueError("only completed finite packed-mixture candidates are accepted")
            shards = phase["shards"]
            if (sum(s["sequences"] for s in shards) != phase["total_sequences"] or
                    sum(s["content_tokens"] for s in shards) != phase["content_tokens"]):
                raise ValueError("candidate sequence/content accounting differs")
            for shard in shards:
                if (shard["sequence_length"] != manifest.config.seq_len or
                        shard["plan_sha256"] != phase["plan_sha256"] or
                        shard["index_offsets_bind_to_sha256"] != shard["index"]["sha256"] or
                        type(shard["valid_targets"]) is not int or
                        not 0 < shard["valid_targets"] <= shard["sequences"]*manifest.config.seq_len):
                    raise ValueError("packed shard geometry, phase or target contract differs")
                for name in ("payload", "index", "index_offsets"):
                    item = shard[name]
                    if item["uri"] not in spec.get("paths", {}) or len(item["sha256"]) != 64:
                        raise ValueError("every packed artifact needs a local path and checksum binding")
    slots = sum(p["total_sequences"] for p in spec["train"])*manifest.config.seq_len
    if slots != manifest.config.total_train_tokens:
        raise ValueError("packed sequence horizon differs from manifest input slots")
    return spec


class PackedTokenBatcher:
    def __init__(self, manifest, *, root=".", rank=0, world_size=1, device=None, verified_files=None):
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid packed rank/world size")
        self.manifest, self.config = manifest, manifest.config
        self.rank, self.world_size, self.device = rank, world_size, device
        self.spec, self.root = stream_spec(manifest), Path(root)
        self.lock = threading.RLock()
        self.ends, self.readers = [], []
        total = 0
        for phase in self.spec["train"]:
            self.readers.append(self.reader(phase, verified_files=verified_files))
            total += phase["total_sequences"]
            self.ends.append(total)

    def reader(self, phase, *, verified_files=None, shards=None):
        mixture = PackedMixture(phase["shards"] if shards is None else shards,
                                seed=phase["seed"], block_sequences=phase["block_sequences"])
        return PackedSequenceReader(mixture, resolve_path=lambda uri: self.root/self.spec["paths"][uri],
                                    verified_files=verified_files)

    def locate(self, index):
        phase = bisect_right(self.ends, index)
        return self.readers[phase], index-(self.ends[phase-1] if phase else 0)

    def indices(self, step):
        start = step*self.config.batch_size*self.world_size
        if start >= self.config.total_examples:
            raise StopIteration
        return [start+i*self.world_size+self.rank for i in range(self.config.batch_size)]

    def plan_update(self, step, micro_steps):
        targets = content = sequences = 0
        with self.lock:
            for micro in range(step, step+micro_steps):
                for index in self.indices(micro):
                    if index < self.config.total_examples:
                        reader, ordinal = self.locate(index)
                        row = reader.describe(ordinal)
                        targets += row["valid_targets"]
                        content += row["content_tokens"]
                        sequences += 1
        return targets, content, sequences*self.config.seq_len

    def train_batch(self, step):
        rows = []
        with self.lock:
            for index in self.indices(step):
                if index < self.config.total_examples:
                    reader, ordinal = self.locate(index)
                    rows.append(reader.sample(ordinal))
                else:
                    length = self.config.seq_len
                    rows.append(dict(inputs=np.full(length, self.manifest.tokenizer.get("pad_token_id", 3)),
                                     targets=np.full(length, -100), positions=np.zeros(length),
                                     segment_ids=np.full(length, -1)))
        return self.batch(rows, self.device)

    @staticmethod
    def batch(rows, device=None):
        inputs, targets = (torch.from_numpy(np.stack([r[name] for r in rows]).astype(np.int64))
                           for name in ("inputs", "targets"))
        layout = DocumentLayout.from_cpu(np.stack([r["positions"] for r in rows]),
                                         np.stack([r["segment_ids"] for r in rows]))
        return (inputs, targets, layout) if device is None else (
            inputs.to(device), targets.to(device), layout.to(device))


def verify_shard(shard, paths, root, vocab_size):
    from production.document_packing import decode_pack

    reader = PackedSequenceReader(PackedMixture([shard], seed="verification"),
                                  resolve_path=lambda uri: Path(root)/paths[uri])
    files, source_counts = {}, Counter()
    payload, metadata, offsets = reader._load(shard)
    counts = dict(content_tokens=0, valid_targets=0, padding_slots=0)
    for start in range(0, len(payload), 1024):
        chunk = payload[start:start+1024]
        if int(chunk.max()) >= vocab_size:
            raise ValueError("packed out-of-vocabulary token")
        if np.any((chunk >= 5) & (chunk <= 31)):
            raise ValueError("reserved vocabulary IDs cannot occur in pretraining content")
    # Sequential metadata traversal avoids permutation/random-I/O during staging.
    try:
        import orjson as json
    except ImportError:
        import json
    for row in range(len(payload)):
        descriptor = json.loads(metadata[int(offsets[row]):int(offsets[row+1])].tobytes())
        targets, _, _ = decode_pack(payload[row], descriptor)
        if int(np.count_nonzero(targets != -100)) != descriptor["valid_targets"]:
            raise ValueError("descriptor target count differs from decoded labels")
        if targets.max() >= vocab_size or np.any((targets < 0) & (targets != -100)):
            raise ValueError("packed lookahead label outside vocabulary")
        for name in counts:
            counts[name] += descriptor[name]
        for segment in descriptor["segments"]:
            if segment["document"].get("kind") == "unlabeled_translation_pair":
                for component in segment["document"]["components"]:
                    source_counts[component["leaf"]] += component["content_tokens"]
            else:
                source_counts[shard["leaf"]] += segment["content_tokens"]
    if any(shard[name] != count for name, count in counts.items()):
        raise ValueError("packed descriptor totals differ from sealed shard")
    for name in ("payload", "index", "index_offsets"):
        item = shard[name]
        path = (Path(root)/paths[item["uri"]]).resolve()
        stat = path.stat()
        files[str(path)] = dict(sha256=item["sha256"], size=stat.st_size, device=stat.st_dev,
                   inode=stat.st_ino, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
    return dict(counts=counts, files=files, source_counts=dict(source_counts))


def verify_shards(shards, paths, root, vocab_size, workers):
    """Bounded CPU processes; pass only three bindings per task, not the corpus catalog."""
    jobs = [(s, {s[name]["uri"]: paths[s[name]["uri"]] for name in
                  ("payload", "index", "index_offsets")}, str(root), vocab_size) for s in shards]
    if workers == 1:
        return [verify_shard(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = [pool.submit(verify_shard, *job) for job in jobs]
        return [future.result() for future in pending]


def validate_packed_manifest(manifest, root, model, *, development_mode=False):
    """Hash artifacts and reconcile every descriptor before admitting a snapshot."""
    from .integrity import fingerprint, manifest_scope
    from .tokenizer import load_tokenizer, tokenizer_contract
    spec = stream_spec(manifest)
    if model.max_seq_len != manifest.config.seq_len or model.pad_token_id != 3 or model.vocab_size > 65536:
        raise ValueError("packed model context/pad identity differs")
    if manifest.tokenizer:
        path = Path(root)/manifest.tokenizer["path"]
        actual = tokenizer_contract(load_tokenizer(path), path)
        for name in ("sha256", "vocab_size", "eos_token_id", "bos_token_id", "pad_token_id", "normalization"):
            if actual[name] != manifest.tokenizer.get(name):
                raise ValueError(f"tokenizer contract mismatch: {name}")
        if actual["vocab_size"] != model.vocab_size or actual["eos_token_id"] != 4:
            raise ValueError("packed vocabulary/EOD identity differs")
    elif not development_mode:
        raise ValueError("packed production requires a frozen tokenizer")
    if not development_mode and (manifest.provenance.get("normalization") != "NFC" or
            not manifest.provenance.get("licenses") or manifest.provenance.get("document_dedup") not in
            {"sha256_nfc", "id_exact_minhash_v1", "id_exact_guard_v1"}):
        raise ValueError("packed production requires preprocessing/dedup/license provenance")
    shards = [s for split in ("train", "val") for phase in spec[split] for s in phase["shards"]]
    scope = manifest_scope(manifest)
    required = set(scope["languages"]) | set(scope["domains"])
    if not development_mode:
        for split in ("train", "val"):
            keys = set()
            for phase in spec[split]:
                phase_keys = {s.get("evaluation_key", s["leaf"].split("/")[0]) for s in phase["shards"]}
                if split == "train" and not required <= phase_keys:
                    raise ValueError("each packed phase needs declared language/domain coverage")
                keys.update(phase_keys)
            if not required <= keys:
                raise ValueError("packed train and heldout splits need declared language/domain coverage")
    if len({s["payload"]["sha256"] for s in shards}) != len(shards):
        raise ValueError("duplicate payload or train/validation contamination")
    if not development_mode and any(phase.get("tokenizer_sha256") != manifest.tokenizer["sha256"]
                                   for split in ("train", "val") for phase in spec[split]):
        raise ValueError("completed phase needs explicit accepted tokenizer lineage")
    workers = int(os.environ.get("CI1_PACKED_VERIFY_WORKERS", "1" if development_mode else str(min(16, os.cpu_count() or 1))))
    if workers <= 0:
        raise ValueError("CI1_PACKED_VERIFY_WORKERS must be positive")
    verification = iter(verify_shards(shards, spec["paths"], root, model.vocab_size, workers))
    files, seen, split_totals, stream_phases = {}, set(), {}, []
    expected_specs = set()
    for split in ("train", "val"):
        split_totals[split] = {}
        for phase in spec[split]:
            if not development_mode and phase.get("tokenizer_sha256") != manifest.tokenizer["sha256"]:
                raise ValueError("completed phase needs explicit accepted tokenizer lineage")
            phase_keys, identities, source_counts = set(), [], Counter()
            for shard in phase["shards"]:
                language = shard["leaf"].split("/")[0]
                key = shard.get("evaluation_key", language)
                phase_keys.add(key)
                identity = shard["payload"]["sha256"]
                if identity in seen:
                    raise ValueError("duplicate payload or train/validation contamination")
                seen.add(identity)
                verified = next(verification)
                counts = verified["counts"]
                source_counts.update(verified["source_counts"])
                files.update(verified["files"])
                split_totals[split][key] = split_totals[split].get(key, 0)+counts["valid_targets"]
                expected_specs.add((spec["paths"][shard["payload"]["uri"]], identity, split,
                                    shard["sequences"]*manifest.config.seq_len, "uint16", shard["leaf"], language))
                identities.append({name: shard[name] for name in
                                   ("leaf", "stratum", "sequences", "valid_targets", "content_tokens") } |
                                  {name: shard[name]["sha256"] for name in ("payload", "index", "index_offsets")})
            if not development_mode and split == "train" and not required <= phase_keys:
                raise ValueError("each packed phase needs declared language/domain coverage")
            if dict(source_counts) != phase["source_content_tokens"]:
                raise ValueError("packed source quotas differ from decoded document/component charges")
            if split == "train":
                stream_phases.append(dict(seed=phase["seed"], block_sequences=phase["block_sequences"],
                                          sequences=phase["total_sequences"], scored_targets=sum(s["valid_targets"] for s in phase["shards"]),
                                          shards=sorted(identities, key=lambda s: (s["payload"], s["index"]))))
        if not development_mode and not required <= split_totals[split].keys():
            raise ValueError("packed train and heldout splits need declared language/domain coverage")
    if any(s.split not in {manifest.config.train_split, manifest.config.val_split} for s in manifest.shards):
        raise ValueError("packed manifest contains an unknown split")
    actual_specs = {(s.path, s.sha256, "train" if s.split == manifest.config.train_split else "val",
                     s.num_tokens, s.dtype, s.source, s.language) for s in manifest.shards}
    if expected_specs != actual_specs or len(actual_specs) != len(manifest.shards):
        raise ValueError("manifest payload catalog differs from packed stream")
    return dict(manifest_sha256=fingerprint(asdict(manifest)), tokenizer=manifest.tokenizer,
                validated_shards=len(seen), tokens_by_key={manifest.config.train_split: split_totals["train"],
                manifest.config.val_split: split_totals["val"]}, scope=scope["name"],
                scored_train_tokens=manifest.scored_train_tokens, input_slots=manifest.config.usable_train_tokens,
                packed_files=files, training_stream=dict(schema="packed_documents_v1", seq_len=manifest.config.seq_len,
                total_examples=manifest.config.total_examples, tokens_before=spec.get("tokens_before", 0), phases=stream_phases))


@torch.no_grad()
def evaluate_packed(model, manifest, root, device, amp_dtype, max_batches, rank, world_size,
                    *, sequences_per_language=None, verified_files=None):
    import math
    import torch.distributed as distributed

    batcher = PackedTokenBatcher(manifest, root=root)
    groups = {}
    for phase in batcher.spec["val"]:
        for key in {s.get("evaluation_key", s["leaf"].split("/")[0]) for s in phase["shards"]}:
            shards = [s for s in phase["shards"] if s.get("evaluation_key", s["leaf"].split("/")[0]) == key]
            groups.setdefault(key, []).append(batcher.reader(phase, shards=shards, verified_files=verified_files))
    result = {}
    for key, readers in sorted(groups.items()):
        ends = np.cumsum([r.mixture.total_sequences for r in readers]).tolist()
        requested = (max_batches*manifest.config.batch_size*world_size
                     if sequences_per_language is None else sequences_per_language)
        selected = min(ends[-1], requested)
        totals = torch.zeros(2, device=device, dtype=torch.float64)
        indices = range(rank, selected, world_size)
        for start in range(0, len(indices), manifest.config.batch_size):
            rows = []
            for index in indices[start:start+manifest.config.batch_size]:
                ordinal = index*(ends[-1]-1)//max(1, selected-1)
                phase = bisect_right(ends, ordinal)
                rows.append(readers[phase].sample(ordinal-(ends[phase-1] if phase else 0)))
            inputs, targets, layout = batcher.batch(rows, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                output = model(inputs, targets, return_logits=False, document_layout=layout)
            totals[0] += output["loss_sum"].double()
            totals[1] += output["loss_tokens"].double()
        if world_size > 1:
            distributed.all_reduce(totals)
        loss_sum, tokens = totals.tolist()
        if tokens <= 0 or not math.isfinite(loss_sum):
            raise FloatingPointError(f"invalid packed heldout evaluation: {key}")
        mean = loss_sum/tokens
        result[key] = dict(loss=mean, tokens=int(tokens), perplexity=math.exp(mean) if mean < 700 else None)
    return result
