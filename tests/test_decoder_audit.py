import json

import pytest

tokenizers = pytest.importorskip('tokenizers')
from tokenizers import Tokenizer, decoders, models, normalizers, Regex

from production.decoder_audit import DecoderAudit, byte_alphabet


@pytest.fixture
def audit(tmp_path):
    # Synthetic shared byte-alphabet/literal vocabulary; no corpus or deployed
    # artifact is copied into Git. Exercises the real Rust decoder behavior.
    pieces = ['<unk>', *[f'<reserved_{n:04d}>' for n in range(1, 32)],
              *byte_alphabet(), *[f'<0x{n:02X}>' for n in range(256)],
              *list(dict.fromkeys('नमस्तेराश्य'))]
    backend = Tokenizer(models.BPE({p: n for n, p in enumerate(pieces)}, [],
                                   unk_token='<unk>', byte_fallback=True))
    backend.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace('—', '-'),
        normalizers.Replace(Regex(r'(?<=\p{Latin})(?=[\p{Devanagari}])'), 'Ġ'),
        normalizers.Replace(' ', 'Ġ'), normalizers.Replace('\n', 'Ċ'),
        normalizers.Replace('\t', 'ĉ'), normalizers.Replace('\r', 'č'),
    ])
    backend.decoder = decoders.Sequence([
        decoders.ByteFallback(), decoders.Fuse(), decoders.ByteLevel(),
        decoders.Replace(Regex('ĠĠ'), ' '), decoders.Replace(Regex('^Ġ'), ''),
        decoders.Replace('Ġ', ' '),
    ])
    path = tmp_path / 'tokenizer.json'
    backend.save(str(path))
    return DecoderAudit(path)


def test_literal_accents_are_corrupted_at_bytelevel_and_ids_are_preserved(audit):
    row = audit.row('accent', 'naïve café', trace=True)
    assert row['stage_trace'][1]['output'] == 'naïveĠcafé'
    assert row['stage_trace'][2]['output'] == 'na�ve caf�'
    assert row['variants']['remove_bytelevel'] == 'naïve café'
    for variant in audit.variants.values():
        assert variant.encode('naïve café', add_special_tokens=False).ids == row['ids']


def test_appending_indic_changes_the_rendering_of_unchanged_prefix_ids(audit):
    result = audit.prefix_change('naïve café', ' नमस्ते')
    assert result['old_decoded'] == 'na�ve caf�'
    assert result['new_decoded'].startswith('naïve café')
    assert not result['old_text_is_prefix_of_new']


def test_marker_encoding_collision_cannot_be_repaired_from_ids_alone(audit):
    result = audit.collision('marker_alias', '\n', 'Ċ')
    assert result['same_ids'] and result['same_routed_stream']
    assert not result['same_policy_text']


def test_removing_bytelevel_fixes_accents_but_breaks_code_whitespace(audit):
    row = audit.row('code', 'def f():\n    return')
    assert row['policy_equal']
    assert row['variants']['remove_bytelevel'] == 'def f():Ċ  return'
    assert row['variants']['literal_utf8_controls'] == 'def f():\n  return'


def test_deliberate_normalization_is_not_counted_as_decoder_corruption(audit):
    row = audit.row('normalization', 'a—b')
    assert not row['raw_equal'] and row['policy_equal']
    assert audit.expected('cafe\u0301') == 'café'
    assert len(byte_alphabet()) == len(set(byte_alphabet())) == 256


def test_complete_audit_keeps_original_artifact_immutable(audit):
    original = audit.path.read_bytes()
    report = audit.run()
    assert audit.path.read_bytes() == original
    assert report['artifact_modified'] is False
    assert report['summary']['fixtures']['policy_roundtrip_failures'] > 0
    assert any(c['same_ids'] and not c['same_policy_text'] for c in report['collisions'])
    assert any(c['same_ids'] and c['same_policy_text'] for c in report['collisions'])
