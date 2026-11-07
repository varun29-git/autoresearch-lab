import hashlib
import json
import struct

import pytest

from production.source_canonicalization import canonicalize_record, convert


class Codec:
    def encode(self, text, *, add_special_tokens=False):
        return type('Encoding', (), {'ids': [ord(c) + 32 for c in text]})()

    def decode(self, ids, *, skip_special_tokens=False):
        return ''.join(chr(i - 32) for i in ids)


def record(text, uid, row='row:0', validation=False):
    return dict(text=text, doc_id=uid, source_uri='s3://source/a', row_identity=row,
        source_text_sha256=hashlib.sha256(text.encode()).hexdigest(), expected_policy_text=text,
        validation=validation, split_family_id=uid)


def test_conversion_preserves_rows_families_and_split_local_offsets(tmp_path):
    val = next(str(i) for i in range(100000) if int.from_bytes(
        hashlib.blake2b(str(i).encode(), digest_size=8).digest(), 'little') % 1000 == 0)
    rows = [record('hello', 'train-family', 'row:0'), record('world', val, 'row:1', validation=True),
            record('again', 'train-family', 'row:2')]
    output = tmp_path / 'converted'
    result = convert(rows, Codec(), output)
    train = [json.loads(line) for line in (output / 'train.index.jsonl').read_text().splitlines()]
    assert [r['token_start'] for r in train] == [0, 5]
    assert [r['row_identity'] for r in train] == ['row:0', 'row:2']
    assert result['content_tokens'] == {'train': 10, 'validation': 5}
    raw = (output / 'validation.tokens.uint16').read_bytes()
    assert Codec().decode(struct.unpack('<5H', raw)) == 'world'
    assert result['production_acceptance'] is False


def test_wrong_source_hash_and_repeated_source_rows_are_rejected(tmp_path):
    row = record('hello', 'family')
    changed = dict(row, text='wrong')
    with pytest.raises(ValueError, match='digest'):
        canonicalize_record(changed, Codec())
    with pytest.raises(ValueError, match='repeats'):
        convert([row, row], Codec(), tmp_path / 'duplicate')


def test_lossy_decoder_is_recorded_without_claiming_acceptance(tmp_path):
    row = record('hello', 'family')
    row['expected_policy_text'] = 'different text'
    result = convert([row], Codec(), tmp_path / 'policy')
    assert result['decoder_policy_mismatch_documents'] == 1
    assert result['production_acceptance'] is False


def test_original_holdout_assignment_is_retained_and_family_leaks_rejected(tmp_path):
    # This identity need not hash into any newly chosen validation predicate.
    row = record('heldout', 'old-validated-family', validation=True)
    _, receipt = canonicalize_record(row, Codec())
    assert receipt['validation'] is True
    other = record('train', 'old-validated-family', 'row:1', validation=False)
    with pytest.raises(ValueError, match='both training and validation'):
        convert([row, other], Codec(), tmp_path / 'leak')
