from pathlib import Path

import pytest

from production.tokenizer_gate import FrozenBatchTokenizer, REGRESSIONS, audit_pairs, compare_encoder, encoder_identity

TOKENIZER = Path(__file__).resolve().parents[1] / 'tokenizer/tokenizer-65k.json'


@pytest.fixture(scope='module')
def batch():
    return FrozenBatchTokenizer(TOKENIZER)


def test_batch_adapter_preserves_frozen_routing_unicode_and_literal_specials(batch):
    assert compare_encoder(batch.tokenizer, batch, REGRESSIONS)['passed']
    result = batch.encode_batch(['naïve café', 'C++में', '<pad>'])
    assert result[0].ids == [44475,227,207,41433,65487,227,201]
    assert result[1].ids == [99,75,75,570]
    assert result[2].ids == [92,144,41494,94]
    assert encoder_identity(TOKENIZER)['normalization'] == 'NFC'
    with pytest.raises(ValueError): batch.encode('hello', add_special_tokens=True)


def test_real_stored_pairs_detect_same_vocabulary_encoding_corruption(batch):
    rows = [{'text': 'C++में', 'ids': [99,4424,570]}, {'text':'<pad>', 'ids':[3]},
            {'text':'naïve café', 'ids':[44475,271,41433,65487,265]}]
    result = audit_pairs(batch.tokenizer, rows)
    assert not result['passed']
    assert result['mismatched_documents'] == 3
    assert result['documents_with_special_ids'] == 1
    correct = [{'text':r['text'],'ids':batch.encode(r['text']).ids} for r in rows]
    assert audit_pairs(batch.tokenizer, correct)['passed']
    assert not audit_pairs(batch.tokenizer, [])['passed']
