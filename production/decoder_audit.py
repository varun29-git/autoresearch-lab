"""Read-only HF decoder diagnosis; reports and corpus samples stay outside Git.

Decoder ablations are diagnostic experiments, not replacement tokenizers.
No vocabulary, encoding, stored token IDs or production artifact is modified.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import unicodedata

from production.artifacts import external_path
from terminus.integrity import file_sha256, write_json_atomic


CASES = (
    ('ascii', 'The rain stopped, and the children went outside.'),
    ('accent', 'naïve café'), ('accent', 'é'), ('accent', 'Ã©'),
    ('accent', 'naïve café नमस्ते'), ('accent', 'नमस्ते naïve café'),
    ('symbol', '°C'), ('symbol', '°C नमस्ते'), ('symbol', 'µ = 0.95'),
    ('code', 'def add(a, b):\n    return a + b'),
    ('code', 'नमस्ते\n    return'),
    ('code', 'def f():\n\treturn "नमस्ते"'),
    ('indic', 'नमस्ते'), ('indic', 'தமிழ் மொழி'), ('indic', 'తెలుగు భాష'),
    ('indic', 'ಕನ್ನಡ'), ('indic', 'বাংলা ভাষা'), ('indic', 'ગુજરાતી ભાષા'),
    ('indic', 'मराठी भाषा'), ('mixed', 'English शब्द'),
    ('mixed', 'C++में'), ('mixed', 'C++ में'), ('mixed', 'hello123नमस्ते'),
    ('control', 'राम\nश्याम'), ('control', 'राम\n श्याम'),
    ('control', 'राम\tश्याम'), ('control', 'राम\t श्याम'),
    ('control', 'राम\rश्याम'), ('control', 'राम\r श्याम'),
    ('fallback', '😀 Hello'), ('fallback', '😀 Hello\n    return'),
    ('fallback', '中文\n    return'), ('fallback', 'Ελληνικά café'),
    ('fallback', 'العربية café'), ('fallback', 'മലയാളം\n    return'),
    ('normalization', '“quote” — dash\u200b\ufeff'),
    ('normalization', 'a\u0301'), ('normalization', 'Hello\u00a0world'),
    ('special', "print('<pad>')"), ('special', '<s> Hello </s>'),
)

COLLISION_PAIRS = (
    ('marker_alias', ' ', 'Ġ'), ('marker_alias', '\n', 'Ċ'),
    ('marker_alias', '\t', 'ĉ'), ('marker_alias', '\r', 'č'),
    ('marker_alias', 'Hello नमस्ते', 'HelloĠनमस्ते'),
    ('marker_alias', 'राम\nश्याम', 'रामĊश्याम'),
    ('control_boundary', 'राम\tश्याम', 'राम\t श्याम'),
    ('control_boundary', 'राम\rश्याम', 'राम\r श्याम'),
    ('control_boundary', 'राम\nश्याम', 'राम\n श्याम'),
    ('intentional_normalization', 'cafe\u0301', 'café'),
    ('intentional_normalization', 'a—b', 'a-b'),
    ('distinct_recoverable', 'C++में', 'C++ में'),
)


def byte_alphabet():
    direct = [*range(33, 127), *range(161, 173), *range(174, 256)]
    chars = list(direct)
    for byte in range(256):
        if byte not in direct:
            direct.append(byte)
            chars.append(256 + sum(c >= 256 for c in chars))
    return [chr(c) for c in chars]


class DecoderAudit:
    def __init__(self, path):
        import tokenizers
        from tokenizers import Tokenizer
        self.path = Path(path)
        self.payload = json.loads(self.path.read_text())
        self.tokenizer = Tokenizer.from_str(json.dumps(self.payload))
        self.version = tokenizers.__version__
        decoder = self.payload.get('decoder')
        if not isinstance(decoder, dict) or decoder.get('type') != 'Sequence':
            raise ValueError('this audit requires an explicit decoder Sequence')
        self.specs = decoder['decoders']
        self.prefixes = []
        for end in range(1, len(self.specs) + 1):
            self.prefixes.append(self._clone(self.specs[:end]))
        self.variants = {
            'remove_fuse': self._clone([s for s in self.specs if s['type'] != 'Fuse']),
            'remove_bytelevel': self._clone([s for s in self.specs if s['type'] != 'ByteLevel']),
        }
        # A diagnostic convention: literals remain UTF-8, controls are restored
        # explicitly, while the original space replacements are retained. Its
        # whitespace failures are measured; this is NOT a promoted decoder.
        literal = [s for s in self.specs if s['type'] != 'ByteLevel']
        controls = [{'type': 'Replace', 'pattern': {'String': a}, 'content': b}
                    for a, b in [('Ċ', '\n'), ('ĉ', '\t'), ('č', '\r')]]
        self.variants['literal_utf8_controls'] = self._clone(literal[:2] + controls + literal[2:])
        self.policy = None
        normalizers = self.payload.get('normalizer', {}).get('normalizers', [])
        for index, normalizer in enumerate(normalizers):
            # This precise boundary is present in the deployed artifact: NFC,
            # quote/dash/space folds precede router/marker transformations.
            pattern = normalizer.get('pattern', {}).get('Regex', '')
            if pattern.startswith(r'(?<=\p{Latin}'):
                clone = deepcopy(self.payload)
                clone['normalizer'] = {'type': 'Sequence', 'normalizers': normalizers[:index]}
                self.policy = Tokenizer.from_str(json.dumps(clone)).normalizer
                self.policy_prefix_count = index
                break

    def _clone(self, specs):
        from tokenizers import Tokenizer
        clone = deepcopy(self.payload)
        clone['decoder'] = {'type': 'Sequence', 'decoders': specs}
        return Tokenizer.from_str(json.dumps(clone))

    def expected(self, text):
        return self.policy.normalize_str(text) if self.policy is not None else text

    def row(self, category, text, *, trace=False):
        encoding = self.tokenizer.encode(text, add_special_tokens=False)
        decoded = self.tokenizer.decode(encoding.ids, skip_special_tokens=False)
        result = {
            'category': category, 'input': text, 'expected_policy_text': self.expected(text),
            'ids': encoding.ids, 'pieces': encoding.tokens, 'decoded': decoded,
            'raw_equal': decoded == text, 'policy_equal': decoded == self.expected(text),
            'normalized_routed_stream': self.tokenizer.normalizer.normalize_str(text),
            'special_ids': sorted({i for i in encoding.ids if i < 32}),
            'variants': {name: t.decode(encoding.ids, skip_special_tokens=False)
                         for name, t in self.variants.items()},
        }
        if trace:
            result['stage_trace'] = [
                {'stage': index + 1, 'decoder': spec, 'output': t.decode(encoding.ids, skip_special_tokens=False)}
                for index, (spec, t) in enumerate(zip(self.specs, self.prefixes))
            ]
            result['skip_special_true'] = self.tokenizer.decode(encoding.ids, skip_special_tokens=True)
        return result

    def collision(self, kind, left, right):
        a = self.tokenizer.encode(left, add_special_tokens=False)
        b = self.tokenizer.encode(right, add_special_tokens=False)
        return {'kind': kind, 'left': left, 'right': right, 'left_ids': a.ids, 'right_ids': b.ids,
                'same_ids': a.ids == b.ids, 'same_policy_text': self.expected(left) == self.expected(right),
                'same_routed_stream': self.tokenizer.normalizer.normalize_str(left) ==
                                      self.tokenizer.normalizer.normalize_str(right)}

    def prefix_change(self, text, continuation):
        prefix = self.tokenizer.encode(text, add_special_tokens=False).ids
        tail = self.tokenizer.encode(continuation, add_special_tokens=False).ids
        old = self.tokenizer.decode(prefix, skip_special_tokens=False)
        new = self.tokenizer.decode(prefix + tail, skip_special_tokens=False)
        return {'prefix_input': text, 'continuation_input': continuation, 'prefix_ids': prefix,
                'appended_ids': tail, 'old_decoded': old, 'new_decoded': new,
                'old_text_is_prefix_of_new': new.startswith(old),
                'scope': 'same existing IDs with appended IDs; no prefix re-encoding'}

    def run(self):
        rows = [self.row(label, text, trace=True) for label, text in CASES]
        matrix = []
        for char in byte_alphabet():
            if ord(char) < 128:
                continue
            for text in (char, f'A {char} Z', f'{char} नमस्ते'):
                row = self.row('byte_alphabet_literal', text)
                row['probe_character'] = char
                row['unicode_name'] = unicodedata.name(char, 'UNNAMED')
                matrix.append(row)
        spacing = [self.row('spacing', prefix + ' ' * count + 'श्याम')
                   for prefix in ('', 'Hello', 'राम') for count in range(1, 9)]
        collisions = [self.collision(*pair) for pair in COLLISION_PAIRS]
        return {
            'schema': 1, 'kind': 'hf_decoder_diagnosis', 'tokenizer_sha256': file_sha256(self.path),
            'tokenizers_version': self.version, 'decoder': self.payload['decoder'],
            'policy_normalizer_prefix_count': getattr(self, 'policy_prefix_count', None),
            'fixtures': rows, 'byte_alphabet_matrix': matrix, 'spacing_matrix': spacing,
            'collisions': collisions,
            'prefix_changes': [self.prefix_change('naïve café', ' नमस्ते'),
                               self.prefix_change('def f():\n    return', ' नमस्ते')],
            'summary': {name: summarize(group) for name, group in
                        [('fixtures', rows), ('byte_alphabet_matrix', matrix), ('spacing_matrix', spacing)]},
            'scope': 'Targeted synthetic diagnosis; no corpus-wide defect-rate or retokenization requirement inferred',
            'artifact_modified': False,
        }


def summarize(rows):
    return {'cases': len(rows), 'raw_roundtrip_failures': sum(not r['raw_equal'] for r in rows),
            'policy_roundtrip_failures': sum(not r['policy_equal'] for r in rows),
            'variant_policy_failures': {name: sum(r['variants'][name] != r['expected_policy_text'] for r in rows)
                                        for name in rows[0]['variants']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True, help='Exact deployed HF tokenizer.json')
    parser.add_argument('--output', required=True, help='External diagnostic report')
    args = parser.parse_args()
    path = external_path(args.tokenizer)
    result = DecoderAudit(path).run()
    output = external_path(args.output, fresh=True)
    write_json_atomic(output, result)
    print(json.dumps({'report': str(output), 'summary': result['summary']}, indent=2))


if __name__ == '__main__':
    main()
