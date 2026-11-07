"""Audit an encoder or source/ID pairs against the frozen production tokenizer.

A same-vocabulary HF JSON is not proof of encoding compatibility. Reports and
source samples stay outside Git. This command never starts cloud tokenization.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from production.artifacts import external_path
from terminus.integrity import file_sha256, fingerprint, write_json_atomic
from terminus.tokenizer import encode_text, normalize_text
from terminus.tokenizer65k import Frozen65kTokenizer


REGRESSIONS = (
    'Hello world.', 'नमस्ते', 'हिन्दी', 'ಕನ್ನಡ', 'தமிழ் மொழி', 'తెలుగు భాష',
    'বাংলা ভাষা', 'ગુજરાતી ભાષા', 'मराठी भाषा', 'C++में', '\\frac{1}{2}',
    '\n    return', 'राम\nश्याम', 'def add(a, b):\n    return a + b',
    '“quote” — dash\u200b\ufeff', '1. తెలుగు', 'नमस्ते world', 'a != b && c == d',
    '<pad>', '<eod>', '<s>', 'naïve café', '😀 Hello', 'a\u0301', 'Hello\u00a0world',
)


@dataclass(frozen=True)
class Encoding:
    ids: list[int]


class FrozenBatchTokenizer:
    """Exact frozen semantics with a minimal batch API; no stock-HF conversion.

    NFC only. Literal special spellings remain ordinary text. The caller adds
    EOD at document boundaries. Other HF features are deliberately unsupported.
    Parallelize independent batches/processes; the Python encoder is not a
    claim of Rust tokenizer performance.
    """
    def __init__(self, path):
        self.tokenizer = Frozen65kTokenizer(path)

    @classmethod
    def from_file(cls, path):
        return cls(path)

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError('add document separators explicitly after content encoding')
        return Encoding(encode_text(self.tokenizer, text))

    def encode_batch(self, texts, *, add_special_tokens=False):
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode(self, ids, *, skip_special_tokens=False):
        if skip_special_tokens:
            raise ValueError('do not silently discard special IDs in a content audit')
        return self.tokenizer.decode(ids)


def encoder_identity(tokenizer_path):
    source = Path(__file__).resolve().parents[1] / 'terminus'
    identity = {'schema': 1, 'encoder': 'Frozen65kTokenizer',
                'tokenizer_sha256': file_sha256(tokenizer_path), 'normalization': 'NFC',
                'special_policy': 'literal_text_no_automatic_special_ids',
                'implementation': {name: file_sha256(source / name)
                                   for name in ('tokenizer.py', 'tokenizer65k.py',
                                                'unicode_contract.py', 'hashing.py')}}
    return identity | {'identity_sha256': fingerprint(identity)}


def compare_encoder(reference, candidate, texts):
    failures = []; count = 0
    for text in texts:
        count += 1
        normalized = normalize_text(text)
        expected = encode_text(reference, text)
        try:
            actual = candidate.encode(text, add_special_tokens=False).ids
            decoded = candidate.decode(actual, skip_special_tokens=False)
            if actual != expected or decoded != normalized or any(i < 32 for i in actual):
                failures.append({'sample': count, 'encoding_equal': actual == expected,
                                 'roundtrip_equal': decoded == normalized,
                                 'special_ids': sorted({i for i in actual if i < 32})})
        except (ValueError, UnicodeError, TypeError) as exc:
            failures.append({'sample': count, 'error': type(exc).__name__})
    return {'samples': count, 'failures': failures, 'passed': count > 0 and not failures}


def audit_pairs(reference, records):
    """Each JSONL row contains source text and the actual stored content IDs."""
    checked = failures = special_documents = 0
    examples = []
    for record in records:
        checked += 1
        ids = record['ids']
        if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < 65536 for i in ids)):
            raise ValueError(f'invalid stored token IDs at record {checked}')
        expected = encode_text(reference, record['text'])
        special_documents += any(i < 32 for i in ids)
        if ids != expected:
            failures += 1
            if len(examples) < 32:
                examples.append({'record': checked, 'expected_tokens': len(expected), 'actual_tokens': len(ids)})
    return {'checked_documents': checked, 'mismatched_documents': failures,
            'documents_with_special_ids': special_documents, 'examples': examples,
            'passed': checked > 0 and failures == 0,
            'scope': 'only supplied source/ID pairs; no claim about unexamined corpus'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--candidate-hf', help='Optional stock HF tokenizer JSON to check, not to approve automatically')
    p.add_argument('--source-id-pairs', help='External JSONL rows with text and actual stored content ids')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = external_path(args.output, fresh=True)
    reference = Frozen65kTokenizer(args.tokenizer)
    candidate = FrozenBatchTokenizer(args.tokenizer)
    if args.candidate_hf:
        from tokenizers import Tokenizer
        candidate = Tokenizer.from_file(str(external_path(args.candidate_hf)))
    result = {'schema': 1, 'kind': 'ci1_tokenizer_compatibility', 'identity': encoder_identity(args.tokenizer),
              'regressions': compare_encoder(reference, candidate, REGRESSIONS)}
    if args.candidate_hf:
        result['candidate_sha256'] = file_sha256(args.candidate_hf)
    if args.source_id_pairs:
        source = external_path(args.source_id_pairs)
        with source.open() as handle:
            result['stored_pairs'] = audit_pairs(reference, (json.loads(line) for line in handle))
        result['source_pairs_sha256'] = file_sha256(source)
    result['passed'] = result['regressions']['passed'] and result.get('stored_pairs', {'passed': True})['passed']
    write_json_atomic(output, result)
    print(json.dumps({'passed': result['passed'], 'report': str(output)}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
