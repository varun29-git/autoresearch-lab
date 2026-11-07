"""Build an external, encoder-preserving decoder candidate for the stock export.

This is not a production tokenizer promotion. Existing IDs, normalization,
merges, special tokens and historical checkpoint identities remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from production.artifacts import external_path
from production.decoder_audit import CASES, DecoderAudit, byte_alphabet
from terminus.integrity import file_sha256, fingerprint, write_json_atomic


INDIC = r'[\p{Devanagari}\p{Bengali}\p{Gujarati}\p{Tamil}\p{Telugu}\p{Kannada}]'
LATIN = r'[\p{Latin}&&[^ĠĊĉč]]'


def router_normalizers():
    """Exact supported router contract, excluding preceding semantic folds."""
    rules = [
        (r'(?<=\p{Latin}[0-9\p{P}\p{S}]*)(?=' + INDIC + ')', 'Ġ'),
        (r'(?<=[\n\r\t])(?=' + INDIC + ')', 'Ġ'),
        ('(?<=' + INDIC + ') (?=' + INDIC + ')', 'Ġ'),
        (r'(?<=\p{Latin}[0-9\p{P}\p{S}]*) (?=' + INDIC + ')', 'ĠĠ'),
        ('(?<= ) (?=' + INDIC + ')', 'ĠĠ'),
        (r'^(?=[^\p{Latin}\s])(?=[^\p{Latin}]*' + INDIC + ')', 'Ġ'),
        (r'^(?=[\p{N}\p{P}\p{S}\p{Z}\p{M}]*' + INDIC + ')', 'Ġ'),
    ]
    return [{'type': 'Replace', 'pattern': {'Regex': pattern}, 'content': content}
            for pattern, content in rules] + [
                {'type': 'Replace', 'pattern': {'String': a}, 'content': b}
                for a, b in [(' ', 'Ġ'), ('\n', 'Ċ'), ('\r', 'č'), ('\t', 'ĉ')]]


def repaired_decoder():
    # One pass removes at most one synthetic marker from each original run.
    # Marker characters must be excluded from Latin in the lookbehind: Ġ is
    # itself a Latin letter and would otherwise consume genuine indentation.
    # A literal indentation run before Greek/Chinese letters is not an Indic
    # prefix. Also stop at internal doubled markers: these were introduced by
    # earlier router rules and prevented either start-of-line insertion.
    # The router's lookahead can cross source newlines. Ċ represents those
    # newlines but belongs to Unicode Latin; explicitly include it here or
    # synthetic prefixes survive on Markdown separators/table/math lines.
    # A genuine leading space can collide with that synthetic prefix in the
    # original encoder. Prefer the unindented representative of that collision;
    # decoder changes cannot recover which spelling the source used.
    line_prefix = (r'(?:(?:Ġ*(?=[\p{N}\p{P}\p{S}\p{Z}\p{M}]|' + INDIC + r'))|'
                   r'(?=[^\p{Latin}Ġ\t\r]))(?:[^\p{Latin}Ġ\t\r]|Ġ(?!Ġ))*' + INDIC)
    # A cross-line dummy precedes an originally unindented symbol/script.
    # Do not apply this extension when another leading marker follows it:
    # those can be real code indentation and must not lose a space.
    crossline_prefix = (r'(?=[^\p{Latin}Ġ\t\r])(?:Ċ|[^\p{Latin}Ġ\t\r]|'
                        r'Ġ+(?=[\p{N}\p{P}\p{S}\p{Z}\p{M}Ċ])|Ġ(?!Ġ))*' + INDIC)
    remove_dummy = (
        r'(?:\A|(?<=Ċ))Ġ(?=(?:' + line_prefix + '|' + crossline_prefix + r'))|'
        r'(?<=' + LATIN + r'[0-9\p{P}\p{S}]*)Ġ(?=Ġ*' + INDIC + r')|'
        r'(?<=[Ċ\t\r])Ġ(?=Ġ*' + INDIC + r')|'
        r'(?<!Ġ)Ġ(?=Ġ+' + INDIC + ')'
    )
    return {'type': 'Sequence', 'decoders': [
        {'type': 'ByteFallback'}, {'type': 'Fuse'},
        # Exactly two markers before Indic distinguish literal ĉ/č followed
        # by one source space from a real tab/CR: the real-control router
        # produces one marker for zero/one space, and >=3 for >=2 spaces.
        *[{'type': 'Replace', 'pattern': {'Regex': a + '(?!ĠĠ' + INDIC + ')'}, 'content': b}
          for a, b in [('ĉ', '\t'), ('č', '\r')]],
        {'type': 'Replace', 'pattern': {'Regex': remove_dummy}, 'content': ''},
        *[{'type': 'Replace', 'pattern': {'String': a}, 'content': b}
          for a, b in [('Ġ', ' '), ('Ċ', '\n')]],
    ]}


def encoder_payload(payload):
    return {key: value for key, value in payload.items() if key != 'decoder'}


def candidate_payload(payload):
    normalizer = payload.get('normalizer') or {}
    if (normalizer.get('type') != 'Sequence'
            or normalizer.get('normalizers', [])[-11:] != router_normalizers()):
        raise ValueError('unsupported router: candidate requires the reviewed stock marker grammar')
    model = payload.get('model') or {}
    if (model.get('type') != 'BPE' or model.get('byte_fallback') is not True
            or model.get('dropout') is not None):
        raise ValueError('candidate requires deterministic BPE with byte fallback')
    if (payload.get('pre_tokenizer') or {}).get('type') != 'Split':
        raise ValueError('unsupported pre-tokenizer for the stock literal-Unicode grammar')
    vocab = model.get('vocab') or {}
    if not all(piece in vocab for piece in ['Ġ', 'Ċ', 'ĉ', 'č', 'न']):
        raise ValueError('required markers/Indic probe are absent from the vocabulary')
    decoder = payload.get('decoder') or {}
    if decoder.get('type') != 'Sequence' or [s.get('type') for s in decoder.get('decoders', [])] != [
            'ByteFallback', 'Fuse', 'ByteLevel', 'Replace', 'Replace', 'Replace']:
        raise ValueError('candidate requires the original stock decoder sequence')
    result = deepcopy(payload)
    result['decoder'] = repaired_decoder()
    assert encoder_payload(result) == encoder_payload(payload)
    return result


class ContextualDecodeStream:
    """Conservative streaming reference for this exact decoder candidate.

    Keeps earlier context that the stock DecodeStream discards. Pending marker
    decisions are checked with Indic continuations after zero, one and two
    markers. Those cover prefix removal and the literal tab/CR alias rule in
    this fixed regex. Emit their common prefix, retaining undecided text.
    Keep an entire trailing byte-fallback run
    until it closes: invalid UTF-8 later in that run can change earlier output.

    This reference decodes retained history and is deliberately not advertised
    as a high-throughput serving implementation. Call finish() once at EOS.
    """
    def __init__(self, tokenizer, *, skip_special_tokens=False):
        decoder = tokenizer.decoder
        if decoder is None or json.loads(decoder.__getstate__()) != repaired_decoder():
            raise ValueError('stream adapter requires the exact candidate decoder')
        indic, space = tokenizer.token_to_id('न'), tokenizer.token_to_id('Ġ')
        if indic is None or space is None:
            raise ValueError('stream probes are absent from the vocabulary')
        self.tokenizer = tokenizer
        dash, newline = tokenizer.token_to_id('-'), tokenizer.token_to_id('Ċ')
        if dash is None or newline is None:
            raise ValueError('cross-line stream probes are absent from the vocabulary')
        # A pending prefix can become synthetic after whitespace closes with
        # punctuation or a newline, not only after an immediate Indic token.
        # Include those lookahead classes before emitting an earlier space.
        self.probes = tuple([space] * count + suffix
                            for count in range(3)
                            for suffix in ([indic], [dash, indic], [newline, indic]))
        self.special = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
        self.skip_special_tokens = skip_special_tokens
        self.ids = []
        self.emitted = ''
        self.closed = False

    def step(self, token_id):
        if self.closed:
            raise RuntimeError('stream has already finished')
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError('token ID must be an integer')
        piece = self.tokenizer.id_to_token(token_id)
        if piece is None:
            raise ValueError('token ID is absent from the vocabulary')
        if self.skip_special_tokens and token_id in self.special:
            return None
        self.ids.append(token_id)
        if re.fullmatch(r'<0x[0-9A-F]{2}>', piece):
            return None
        text = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        end = len(text)
        for suffix in self.probes:
            future = self.tokenizer.decode(self.ids + suffix, skip_special_tokens=False)
            end = min(end, next((i for i, (a, b) in enumerate(zip(text, future)) if a != b),
                                min(len(text), len(future))))
        return self._emit(text[:end])

    def _emit(self, text):
        if not text.startswith(self.emitted):
            raise RuntimeError('candidate streaming prefix changed; stop emitting output')
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta or None

    def finish(self):
        if self.closed:
            raise RuntimeError('stream has already finished')
        text = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        delta = self._emit(text)
        self.closed = True
        return delta or ''


def qualification_cases():
    rows = list(CASES)
    for char in byte_alphabet():
        if ord(char) >= 128:
            rows.extend(('byte_alphabet_literal', text)
                        for text in (char, f'A {char} Z', f'{char} नमस्ते'))
    rows.extend(('spacing', prefix + ' ' * count + 'श्याम')
                for prefix in ('', 'Hello', 'राम') for count in range(1, 9))
    parts = ['राम', 'C++', 'Hello123', '中文', 'Ελληνικά', '😀', '42', '(', '-', 'தமிழ்', 'বাংলা']
    separators = ['', ' ', '  ', '   ', '\n', '\n ', '\n    ', '\t', '\t ', '\r', '\r ',
                  '\r\n', '\r\n  ']
    rows.extend(('boundary_matrix', left + separator + right)
                for left in parts for separator in separators for right in parts)
    rows.extend(('leading_indent_matrix', head + ' ' * count + first + gap + 'श्याम')
                for head in ['', 'राम\n'] for count in range(5)
                for first in ['中文', 'Ελληνικά', '(', '(-', '😀'] for gap in ['', ' ', '  ', '   '])
    return rows


def qualify(audit, candidate, *, extra_cases=()):
    rows = []
    for category, text in [*qualification_cases(), *extra_cases]:
        original = audit.tokenizer.encode(text, add_special_tokens=False).ids
        ids = candidate.encode(text, add_special_tokens=False).ids
        decoded = candidate.decode(ids, skip_special_tokens=False)
        canonical_ids = candidate.encode(decoded, add_special_tokens=False).ids
        stream = ContextualDecodeStream(candidate)
        chunks = [chunk for token_id in ids if (chunk := stream.step(token_id)) is not None]
        chunks.append(stream.finish())
        rows.append({'category': category, 'input': text, 'ids': ids,
                     'expected': audit.expected(text), 'decoded': decoded,
                     'original_decoded': audit.tokenizer.decode(original, skip_special_tokens=False),
                     'encoding_equal': original == ids, 'policy_equal': decoded == audit.expected(text),
                     'decoded_reencodes_same_ids': canonical_ids == ids,
                     'stream_equal': ''.join(chunks) == decoded})
    collisions = [audit.collision(*pair) for pair in [
        ('leading_space', '中文 नमस्ते', ' 中文 नमस्ते'),
        ('leading_space', 'Ελληνικά नमस्ते', ' Ελληνικά नमस्ते'),
    ]]
    return {'rows': rows, 'additional_collision_probes': collisions,
            'summary': {'cases': len(rows),
                        'encoder_id_mismatches': sum(not r['encoding_equal'] for r in rows),
                        'original_policy_failures': sum(r['original_decoded'] != r['expected'] for r in rows),
                        'candidate_policy_failures': sum(not r['policy_equal'] for r in rows),
                        'same_id_alternative_outputs': sum(not r['policy_equal'] and
                                                          r['decoded_reencodes_same_ids'] for r in rows),
                        'contextual_stream_mismatches': sum(not r['stream_equal'] for r in rows)}}


def build_candidate(source, output_dir, *, extra_cases=()):
    from tokenizers import Tokenizer
    source = external_path(source)
    output_dir = external_path(output_dir, fresh=True)
    if source.parent == output_dir or source.parent in output_dir.parents:
        raise ValueError('candidate must be separate from the source tokenizer directory')
    original_sha = file_sha256(source)
    audit = DecoderAudit(source)
    payload = candidate_payload(audit.payload)
    candidate = Tokenizer.from_str(json.dumps(payload))
    evidence = qualify(audit, candidate, extra_cases=extra_cases)
    if evidence['summary']['encoder_id_mismatches'] or evidence['summary']['contextual_stream_mismatches']:
        raise ValueError('candidate failed encoder/stream consistency; do not export')
    if file_sha256(source) != original_sha:
        raise ValueError('source tokenizer changed during qualification')
    # Serialization changes the file hash. Bind lineage rather than pretending
    # the candidate matches an existing checkpoint's full-artifact hash.
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / 'tokenizer.json'
    write_json_atomic(path, payload)
    auxiliary = {}
    for name in ['tokenizer_config.json', 'special_tokens_map.json']:
        original = source.parent / name
        if original.is_file():
            (output_dir / name).write_bytes(original.read_bytes())
            auxiliary[name] = file_sha256(output_dir / name)
    receipt = {'schema': 1, 'kind': 'encoder_preserving_decoder_candidate',
               'source_sha256': original_sha, 'candidate_sha256': file_sha256(path),
               'encoder_payload_sha256': fingerprint(encoder_payload(payload)),
               'encoder_payload_equal': encoder_payload(audit.payload) == encoder_payload(payload),
               'tokenizers_version': audit.version, 'auxiliary_sha256': auxiliary,
               'production_accepted': False,
               'limits': ['targeted CPU fixtures, not corpus-wide acceptance',
                          'identical-ID collisions cannot be universally inverted',
                          'stock DecodeStream and actual serving adapters need separate qualification',
                          'ContextualDecodeStream is a conservative retained-history reference',
                          'candidate full hash must not replace historical checkpoint identity silently'],
               'qualification': evidence}
    write_json_atomic(output_dir / 'qualification.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output-dir', required=True, help='Fresh external candidate directory')
    parser.add_argument('--source-samples-jsonl', help='Optional external original-text samples with text fields')
    args = parser.parse_args()
    samples = []
    if args.source_samples_jsonl:
        with external_path(args.source_samples_jsonl).open() as handle:
            for number, line in enumerate(handle, 1):
                text = json.loads(line).get('text')
                if not isinstance(text, str):
                    raise ValueError(f'sample line {number} needs a string text field')
                samples.append(('original_source_sample', text))
    receipt = build_candidate(args.tokenizer, args.output_dir, extra_cases=samples)
    print(json.dumps({'candidate': str(Path(args.output_dir) / 'tokenizer.json'),
                      'production_accepted': receipt['production_accepted'],
                      'summary': receipt['qualification']['summary']}, indent=2))


if __name__ == '__main__':
    main()
