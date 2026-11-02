import json
import random

import pytest

pytest.importorskip('tokenizers')
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
from tokenizers.decoders import DecodeStream

from production.decoder_audit import byte_alphabet
from production.decoder_repair import (ContextualDecodeStream, build_candidate, candidate_payload,
                                       encoder_payload, router_normalizers)


@pytest.fixture
def codec(tmp_path):
    # Synthetic literal-Unicode vocabulary with the deployed router grammar.
    # No real corpus, generated result or production tokenizer enters Git.
    specials = ['<unk>', *[f'<reserved_{n:04d}>' for n in range(1, 32)]]
    specials[3] = '<pad>'
    alphabet = 'नमस्तेराश्यमेंதமிழ்மொழితెలుగుభాషಕನ್ನಡবাংলাভাষાગુજરાતીમરાઠी'
    pieces = [*specials, *byte_alphabet(), *[f'<0x{n:02X}>' for n in range(256)]]
    pieces.extend(c for c in dict.fromkeys(alphabet) if c not in pieces)
    original = Tokenizer(models.BPE({p: n for n, p in enumerate(pieces)}, [],
                                    unk_token='<unk>', byte_fallback=True))
    original.add_special_tokens([AddedToken('<pad>', special=True)])
    payload = json.loads(original.to_str())
    payload['normalizer'] = {'type': 'Sequence', 'normalizers': [
        {'type': 'NFC'}, {'type': 'Replace', 'pattern': {'String': '—'}, 'content': '-'},
        *router_normalizers(),
    ]}
    original = Tokenizer.from_str(json.dumps(payload))
    original.pre_tokenizer = pre_tokenizers.Split(Regex(r'[\s\S]'), behavior='isolated')
    original.decoder = decoders.Sequence([
        decoders.ByteFallback(), decoders.Fuse(), decoders.ByteLevel(),
        decoders.Replace(Regex('ĠĠ'), ' '), decoders.Replace(Regex('^Ġ'), ''),
        decoders.Replace('Ġ', ' '),
    ])
    source = tmp_path / 'source' / 'tokenizer.json'
    source.parent.mkdir()
    original.save(str(source))
    candidate = Tokenizer.from_str(json.dumps(candidate_payload(json.loads(original.to_str()))))
    return original, candidate, source


@pytest.mark.parametrize('text', [
    'naïve café', 'Ã©', '°C', 'µ = 0.95', 'naïve café नमस्ते',
    'def add(a, b):\n    return a + b', 'नमस्ते\n    return',
    'def f():\n\treturn "नमस्ते"', 'राम\nश्याम', 'राम\n श्याम',
    'C++में', 'C++ में', 'hello123नमस्ते', 'Hello शब्द',
    'தமிழ் மொழி', 'తెలుగు భాష', 'ಕನ್ನಡ', 'বাংলা ভাষা', 'ગુજરાતી', 'मराठी भाषा',
    '😀 Hello\n    return', '中文\n    return', 'Ελληνικά café',
    'العربية café', 'മലയാളം\n    return',
    'ĉ नमस्ते', 'č नमस्ते',
])
def test_unicode_code_and_recoverable_boundaries(codec, text):
    original, candidate, _ = codec
    ids = original.encode(text, add_special_tokens=False).ids
    assert candidate.encode(text, add_special_tokens=False).ids == ids
    assert candidate.decode(ids, skip_special_tokens=False) == text


def test_indentation_and_real_space_runs_are_preserved(codec):
    _, candidate, _ = codec
    for left in ['', 'Hello', 'राम', 'राम\n', 'def f():\n']:
        for right in ['श्याम', 'return', 'café']:
            for count in range(9):
                text = left + ' ' * count + right
                # A zero-space line with no target-script text has no dummy.
                ids = candidate.encode(text, add_special_tokens=False).ids
                assert candidate.decode(ids, skip_special_tokens=False) == text


def test_semantic_folds_and_encoder_payload_remain_unchanged(codec):
    original, candidate, _ = codec
    a, b = json.loads(original.to_str()), json.loads(candidate.to_str())
    assert encoder_payload(a) == encoder_payload(b)
    assert a['decoder'] != b['decoder']
    text = 'cafe\u0301—a'
    ids = candidate.encode(text, add_special_tokens=False).ids
    assert ids == original.encode(text, add_special_tokens=False).ids
    assert candidate.decode(ids, skip_special_tokens=False) == 'café-a'


@pytest.mark.parametrize('text', [
    '  中文श्याम', '   Ελληνικά श्याम', ' 中文  श्याम', ' 😀   श्याम',
    'राम\n   中文 श्याम', 'राम\n  (  श्याम', '    (- श्याम',
])
def test_only_synthetic_prefixes_are_removed_from_complex_indentation(codec, text):
    _, candidate, _ = codec
    ids = candidate.encode(text, add_special_tokens=False).ids
    assert candidate.decode(ids, skip_special_tokens=False) == text
    assert stream_text(candidate, ids) == text


def test_irrecoverable_collisions_have_explicit_canonical_output(codec):
    _, candidate, _ = codec
    for a, b, canonical in [
        ('\n', 'Ċ', '\n'), ('\t', 'ĉ', '\t'), (' ', 'Ġ', ' '),
        ('राम\tश्याम', 'राम\t श्याम', 'राम\tश्याम'),
        ('中文 नमस्ते', ' 中文 नमस्ते', '中文 नमस्ते'),
    ]:
        ids = candidate.encode(a, add_special_tokens=False).ids
        assert ids == candidate.encode(b, add_special_tokens=False).ids
        assert candidate.decode(ids, skip_special_tokens=False) == canonical


def stream_text(candidate, ids, *, skip_special_tokens=False):
    stream = ContextualDecodeStream(candidate, skip_special_tokens=skip_special_tokens)
    chunks = [chunk for token_id in ids if (chunk := stream.step(token_id)) is not None]
    chunks.append(stream.finish())
    return ''.join(chunks)


def test_contextual_stream_keeps_the_latin_context_stock_stream_discards(codec):
    _, candidate, _ = codec
    text = 'C++में'
    ids = candidate.encode(text, add_special_tokens=False).ids
    stock = DecodeStream(skip_special_tokens=False)
    try:
        chunks = [chunk for token_id in ids if (chunk := stock.step(candidate, token_id)) is not None]
    except Exception as error:
        assert 'Invalid prefix' in str(error)
    else:
        assert ''.join(chunks) != text
    assert stream_text(candidate, ids) == text


def test_stream_matches_batch_over_context_changes_and_byte_fallback(codec):
    _, candidate, _ = codec
    texts = ['naïve café नमस्ते', 'C++में', 'hello123नमस्ते', 'राम\n    श्याम',
             '中文 नमस्ते', '😀 Hello\n    return', 'नमस्ते\n\treturn',
             '   राम', 'राम   श्याम', "print('<pad>')"]
    for text in texts:
        ids = candidate.encode(text, add_special_tokens=False).ids
        for skip in [False, True]:
            assert stream_text(candidate, ids, skip_special_tokens=skip) == candidate.decode(
                ids, skip_special_tokens=skip)


def test_pending_byte_run_is_not_emitted_before_it_closes(codec):
    _, candidate, _ = codec
    vocab = candidate.get_vocab()
    stream = ContextualDecodeStream(candidate)
    # Later invalid bytes change ByteFallback's interpretation of the whole
    # run. Do not publish an earlier valid character prematurely.
    for token_id in [vocab['<0xC3>'], vocab['<0xA9>'], vocab['<0xFF>']]:
        assert stream.step(token_id) is None
    assert stream.finish() == candidate.decode(stream.ids, skip_special_tokens=False)


def test_skipped_special_does_not_close_a_byte_fallback_run(codec):
    _, candidate, _ = codec
    vocab = candidate.get_vocab()
    ids = [vocab['<0xC3>'], vocab['<pad>'], vocab['<0xA9>']]
    assert stream_text(candidate, ids, skip_special_tokens=True) == 'é'
    assert stream_text(candidate, ids, skip_special_tokens=False) == candidate.decode(
        ids, skip_special_tokens=False)


def test_arbitrary_generated_id_streams_match_batch(codec):
    _, candidate, _ = codec
    vocab = candidate.get_vocab()
    alphabet = [vocab[c] for c in 'AC+()नमĠĊĉčéï']
    alphabet += [vocab[f'<0x{n:02X}>'] for n in [0xC3, 0xA9, 0xFF, 0xE0, 0xA4, 0xA8]]
    rng = random.Random(9127)
    for _ in range(120):
        ids = rng.choices(alphabet, k=rng.randrange(1, 70))
        assert stream_text(candidate, ids) == candidate.decode(ids, skip_special_tokens=False)


def test_stream_rejects_unreviewed_decoder_and_invalid_lifecycle(codec):
    original, candidate, _ = codec
    with pytest.raises(ValueError, match='exact candidate'):
        ContextualDecodeStream(original)
    stream = ContextualDecodeStream(candidate)
    with pytest.raises(ValueError, match='integer'):
        stream.step(True)
    with pytest.raises(ValueError, match='absent'):
        stream.step(999999)
    stream.finish()
    with pytest.raises(RuntimeError, match='finished'):
        stream.step(1)


def test_repair_rejects_changed_router_before_export(codec):
    original, _, _ = codec
    payload = json.loads(original.to_str())
    payload['normalizer']['normalizers'][-1]['content'] = 'different'
    with pytest.raises(ValueError, match='unsupported router'):
        candidate_payload(payload)


def test_external_candidate_export_binds_lineage_and_keeps_source(codec, tmp_path):
    original, _, source = codec
    before = source.read_bytes()
    output = tmp_path / 'candidate'
    receipt = build_candidate(source, output)
    assert source.read_bytes() == before
    assert receipt['encoder_payload_equal']
    assert receipt['source_sha256'] != receipt['candidate_sha256']
    assert receipt['production_accepted'] is False
    assert receipt['qualification']['summary']['encoder_id_mismatches'] == 0
    assert receipt['qualification']['summary']['contextual_stream_mismatches'] == 0
    exported = Tokenizer.from_file(str(output / 'tokenizer.json'))
    assert encoder_payload(json.loads(exported.to_str())) == encoder_payload(json.loads(original.to_str()))
    with pytest.raises(FileExistsError):
        build_candidate(source, output)
