"""Contextual, incremental decoding for the frozen 65K token grammar."""
from __future__ import annotations

import codecs

from .tokenizer65k import (Frozen65kTokenizer, BYTE_BASE, BYTE_FALLBACK_BASE,
                          INDIC_BASE, ENGLISH_TAIL_BASE, SPACE_MARK)


def unused_token_ids(tokenizer):
    return tuple(i for i, piece in enumerate(tokenizer.pieces)
                 if isinstance(piece, str) and piece.startswith("<reserved"))


class FrozenDecodeStream:
    """Preserve Indic prefix context and UTF-8 bytes across individual tokens.

    Prime with prompt tokens before decoding a continuation. A maximum-token
    cutoff may leave an incomplete UTF-8 suffix; it can be dropped explicitly,
    but malformed interior bytes and unexpected control IDs always raise.
    """
    def __init__(self, tokenizer):
        if not isinstance(tokenizer, Frozen65kTokenizer):
            raise TypeError("stream requires the frozen 65K token grammar")
        self.tokenizer = tokenizer
        self.forbidden = set(range(32)) | set(unused_token_ids(tokenizer))
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.previous = None
        self.closed = False
        self.dropped_incomplete_bytes = 0

    def step(self, token):
        if self.closed: raise ValueError("decoder is already closed")
        if type(token) is not int or not 0 <= token < 65536 or token in self.forbidden:
            raise ValueError("unexpected special, reserved or invalid token ID")
        tok = self.tokenizer
        if BYTE_BASE <= token < BYTE_FALLBACK_BASE:
            data = bytes([token - BYTE_BASE])
        elif BYTE_FALLBACK_BASE <= token < INDIC_BASE:
            data = bytes([token - BYTE_FALLBACK_BASE])
        elif token >= ENGLISH_TAIL_BASE:
            data = bytes(tok.byte_of_char[c] for c in tok.pieces[token])
        else:
            piece = tok.pieces[token]
            if piece.startswith(SPACE_MARK) and len(piece) > 1 and self.previous != "indic":
                piece = piece[1:]
            data = piece.replace(SPACE_MARK, " ").encode("utf-8")
        kind = tok._decode_class(token)
        if kind != "neutral": self.previous = kind
        return self.decoder.decode(data, final=False)

    def finish(self, *, allow_incomplete=False):
        if self.closed: raise ValueError("decoder is already closed")
        self.closed = True
        pending = self.decoder.getstate()[0]
        if allow_incomplete and pending:
            self.dropped_incomplete_bytes = len(pending)
            return ""
        return self.decoder.decode(b"", final=True)


def decode_continuation(tokenizer, prompt_ids, continuation_ids, *, stop_ids=(), max_token_cutoff=False):
    ids = list(continuation_ids)
    stopped = bool(ids and ids[-1] in stop_ids)
    if stopped: ids.pop()
    if any(token in stop_ids for token in ids):
        raise ValueError("generation continued after a stop token")
    stream = FrozenDecodeStream(tokenizer)
    for token in prompt_ids: stream.step(token)
    if stream.decoder.getstate()[0]:
        raise ValueError("prompt ends inside a UTF-8 character")
    chunks = [stream.step(token) for token in ids]
    chunks.append(stream.finish(allow_incomplete=max_token_cutoff and not stopped))
    return "".join(chunks), stream.dropped_incomplete_bytes
