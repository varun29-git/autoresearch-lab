"""Bounded best-fit document packing with explicit causal segment metadata.

Long documents are split before packing; packing never splits a fragment.
Input content is stored once. A full fragment uses one label-only lookahead
from the same selected document span. Only true document endings emit EOD.
This CPU format requires a segment-aware training loader and attention path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np

SCHEMA = "terminus_document_packing_v1"


@dataclass
class Fragment:
    tokens: np.ndarray
    document: dict
    offset: int
    document_end: bool
    lookahead: int | None
    eod_input: bool

    @property
    def content_tokens(self):
        return len(self.tokens) - int(self.eod_input)


def fragments(tokens, document, *, length=4096, eod=4, document_end=True,
              offset=0):
    """Yield disjoint input spans; no question repetition or scored overlap.

    document_end=False is a quota boundary whose unused source suffix remains
    reserved. Its last label is masked rather than leaking the next phase.
    """
    values = np.asarray(tokens)
    if length < 2 or values.ndim != 1 or values.dtype.kind not in "ui":
        raise ValueError("invalid document token array or context length")
    if not len(values) or np.any(values >= 65536):
        raise ValueError("empty document or token outside vocabulary")
    if np.any(values < 0) or np.any((values >= 3) & (values <= 31)):
        raise ValueError("source content contains padding, EOD or reserved IDs")
    for start in range(0, len(values), length):
        stop = min(start + length, len(values))
        last = stop == len(values)
        complete = last and document_end
        data = values[start:stop].astype("<u2", copy=True)
        eod_input = complete and len(data) < length
        if eod_input:
            data = np.concatenate((data, np.array([eod], dtype="<u2")))
        lookahead = None if eod_input else (
            int(values[stop]) if not last else eod if complete else None)
        yield Fragment(data, document, offset + start, complete, lookahead,
                       eod_input)


def best_fit(items, length=4096, seed=0):
    """Best-fit-decreasing using integer capacity buckets, O(n log L).

    A Python integer bitset locates the smallest available fitting capacity.
    Payload copying stays in NumPy. Shuffling complete packs removes the
    length order introduced by BFD without changing document contents.
    """
    ordered = sorted(enumerate(items), key=lambda x: (-len(x[1].tokens), x[0]))
    buckets = [[] for _ in range(length + 1)]
    available = 0
    packs = []
    for _, item in ordered:
        size = len(item.tokens)
        if not 0 < size <= length:
            raise ValueError("fragment exceeds context")
        candidates = available >> size
        if candidates:
            remaining = size + ((candidates & -candidates).bit_length() - 1)
            index = buckets[remaining].pop()
            if not buckets[remaining]:
                available &= ~(1 << remaining)
        else:
            index = len(packs)
            packs.append([])
            remaining = length
        packs[index].append(item)
        remaining -= size
        if remaining:
            buckets[remaining].append(index)
            available |= 1 << remaining
    random.Random(seed).shuffle(packs)
    return packs


def encode_pack(items, *, length=4096, pad=3):
    inputs = np.full(length, pad, dtype="<u2")
    segments = []
    cursor = 0
    for item in items:
        end = cursor + len(item.tokens)
        if end > length:
            raise ValueError("overfull sequence")
        inputs[cursor:end] = item.tokens
        segments.append({"start": cursor, "end": end,
            "content_tokens": item.content_tokens,
            "document_offset": item.offset, "document_end": item.document_end,
            "eod_input": item.eod_input, "last_label": item.lookahead,
            "document": item.document})
        cursor = end
    valid = sum(s["end"] - s["start"] - 1 + (s["last_label"] is not None)
                for s in segments)
    descriptor = {"segments": segments, "used_slots": cursor,
        "content_tokens": sum(x.content_tokens for x in items),
        "valid_targets": valid, "padding_slots": length - cursor}
    return inputs, descriptor


def decode_pack(inputs, descriptor, *, pad=3, eod=4):
    """CPU target/position oracle; production must preserve this contract."""
    inputs = np.asarray(inputs)
    targets = np.full(len(inputs), -100, dtype=np.int64)
    positions = np.zeros(len(inputs), dtype=np.int32)
    segments = np.full(len(inputs), -1, dtype=np.int32)
    cursor = 0
    content = 0
    for segment_id, item in enumerate(descriptor["segments"]):
        start, end = item["start"], item["end"]
        if start != cursor or not start < end <= len(inputs):
            raise ValueError("non-contiguous or invalid segment")
        if item["eod_input"] and (not item["document_end"] or inputs[end-1] != eod):
            raise ValueError("EOD differs from document boundary")
        if item["eod_input"] and item["last_label"] is not None:
            raise ValueError("terminal EOD must not predict another document")
        targets[start:end-1] = inputs[start+1:end]
        if item["last_label"] is not None:
            targets[end-1] = item["last_label"]
        positions[start:end] = np.arange(end-start)
        segments[start:end] = segment_id
        content += item["content_tokens"]
        cursor = end
    if (cursor != descriptor["used_slots"] or np.any(inputs[cursor:] != pad)
            or len(inputs)-cursor != descriptor["padding_slots"]
            or content != descriptor["content_tokens"]
            or int(np.count_nonzero(targets != -100)) != descriptor["valid_targets"]):
        raise ValueError("descriptor counters or padding differ from payload")
    return targets, positions, segments


def causal_mask(segment_ids, positions, *, window=None):
    """Small CPU reference only; never construct this dense mask on the GPU.

    window denotes total visible positions including the current position.
    The native adapter must translate that definition to its window API.
    """
    s = np.asarray(segment_ids)
    p = np.asarray(positions)
    distance = p[:, None] - p[None, :]
    mask = (s[:, None] == s[None, :]) & (s[:, None] >= 0) & (distance >= 0)
    if window is not None:
        mask &= distance < window
    return mask


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ShardWriter:
    """Sequential payload plus descriptors; hashes are emitted on close."""
    def __init__(self, directory, name, *, length=4096, audit_stride=1024,
                 buffer_bytes=1<<20, fast_json=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.payload = self.directory / (name + ".uint16")
        self.index = self.directory / (name + ".segments.jsonl")
        self.data = self.payload.open("xb", buffering=buffer_bytes)
        self.meta = self.index.open("xb", buffering=buffer_bytes)
        self.data_hash = hashlib.sha256()
        self.meta_hash = hashlib.sha256()
        self.meta_bytes = 0
        self.fast_json = fast_json
        self.length = length
        self.audit_stride = audit_stride
        self.counts = dict(sequences=0, content_tokens=0, valid_targets=0,
                           used_slots=0, padding_slots=0)

    def add(self, items):
        inputs, descriptor = encode_pack(items, length=self.length)
        if self.counts["sequences"] % self.audit_stride == 0:
            decode_pack(inputs, descriptor)
        payload=inputs.tobytes()
        if self.fast_json:
            import orjson
            metadata=orjson.dumps(descriptor,option=orjson.OPT_APPEND_NEWLINE)
        else:
            metadata=(json.dumps(descriptor,ensure_ascii=False,separators=(",", ":"))+"\n").encode('utf-8')
        self.data.write(payload);self.data_hash.update(payload)
        self.meta.write(metadata);self.meta_hash.update(metadata);self.meta_bytes+=len(metadata)
        self.counts["sequences"] += 1
        for key in self.counts.keys() - {"sequences"}:
            self.counts[key] += descriptor[key]

    def append_encoded(self, payload, metadata, counts):
        if len(payload)!=counts['sequences']*self.length*2:
            raise ValueError('encoded chunk payload size differs')
        self.data.write(payload);self.data_hash.update(payload)
        self.meta.write(metadata);self.meta_hash.update(metadata);self.meta_bytes+=len(metadata)
        for name in self.counts:self.counts[name]+=counts[name]

    def close(self):
        self.data.close()
        self.meta.close()
        return {"schema": SCHEMA, "sequence_length": self.length,
            **self.counts, "payload": {"path": str(self.payload),
                "bytes": self.payload.stat().st_size, "sha256": self.data_hash.hexdigest()},
            "index": {"path": str(self.index), "bytes": self.index.stat().st_size,
                      "sha256": self.meta_hash.hexdigest()},
            "cross_document_mask_required": True, "native_training_accepted": False}
