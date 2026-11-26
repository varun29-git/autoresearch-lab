"""Source-based CPU conversion pilot; never promotes or launches a corpus.

Original source text, its digest, and row/family identities are mandatory.
Decoding old token IDs is deliberately not a source of conversion text.
All resulting records, bins, receipts and configs remain outside Git.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
from pathlib import Path
import sys


def canonicalize_record(record, tokenizer):
    text = record['text']
    if not isinstance(text, str) or not text:
        raise ValueError('original source text must be a nonempty string')
    source = record['source_uri']
    identity = record['row_identity']
    uid = record['doc_id']
    if not all(isinstance(v, str) and v for v in (source, identity, uid)):
        raise ValueError('source URI, original row identity and family ID are required')
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != record['source_text_sha256']:
        raise ValueError('original source text digest differs from its verified receipt')
    expected = record['expected_policy_text']
    if not isinstance(expected, str):
        raise ValueError('explicit semantic normalization policy text is required')
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or not 32 <= i < 65536 for i in ids):
        raise ValueError('canonical content contains empty, reserved or out-of-vocabulary IDs')
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    # Historical writers use different split predicates. Rehashing an old
    # heldout document can silently move it into training; retain the proven
    # source assignment and reconcile shared families before materialization.
    validation = record['validation']
    family = record['split_family_id']
    if not isinstance(validation, bool) or not isinstance(family, str) or not family:
        raise ValueError('original validation membership and split family identity are required')
    return ids, dict(doc_id=uid, row_identity=identity, source_uri=source,
        source_text_sha256=digest, n_tokens=len(ids), validation=validation, split_family_id=family,
        decoded_policy_equal=decoded == expected,
        decode_reencode_equal=tokenizer.encode(decoded, add_special_tokens=False).ids == ids)


def convert(records, tokenizer, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {'train': 0, 'validation': 0}
    documents = bad_policy = bad_reencode = 0
    seen = set()
    families = {}
    from contextlib import ExitStack
    with ExitStack() as stack:
        bins = {split: stack.enter_context((output / (split + '.tokens.uint16')).open('wb')) for split in counts}
        indices = {split: stack.enter_context((output / (split + '.index.jsonl')).open('w')) for split in counts}
        for record in records:
            identity = record['source_uri'], record['row_identity']
            if identity in seen:
                raise ValueError('conversion input repeats an original source row')
            seen.add(identity)
            ids, receipt = canonicalize_record(record, tokenizer)
            family = receipt['split_family_id']
            if family in families and families[family] != receipt['validation']:
                raise ValueError('the same source family appears in both training and validation')
            families[family] = receipt['validation']
            split = 'validation' if receipt['validation'] else 'train'
            receipt['token_start'] = counts[split]
            data = array.array('H', ids)
            if data.itemsize != 2:
                raise ValueError('host does not supply a two-byte unsigned short')
            if sys.byteorder != 'little':
                data.byteswap()
            bins[split].write(data.tobytes())
            indices[split].write(json.dumps(receipt, ensure_ascii=False) + '\n')
            counts[split] += len(ids)
            documents += 1
            bad_policy += not receipt['decoded_policy_equal']
            bad_reencode += not receipt['decode_reencode_equal']
    summary = dict(documents=documents, content_tokens=counts,
        decoder_policy_mismatch_documents=bad_policy,
        decode_reencode_mismatch_documents=bad_reencode,
        production_acceptance=False, original_bins_modified=False,
        source_document_boundaries_preserved=True, eod_inserted=False,
        complete_corpus_converted=False, paid_compute_launched=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    from production.artifacts import external_path
    from terminus.integrity import file_sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--tokenizer-sha256', required=True)
    parser.add_argument('--encoder-kind', required=True, choices=('frozen', 'hf'),
                        help='explicit execution path; a vocabulary hash cannot select it')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    tokenizer_path = Path(args.tokenizer).resolve()
    if file_sha256(tokenizer_path) != args.tokenizer_sha256:
        raise ValueError('canonical tokenizer differs from the explicitly pinned artifact')
    input_path = external_path(args.records)
    records = (json.loads(line) for line in input_path.read_text().splitlines() if line)
    if args.encoder_kind == 'frozen':
        from production.tokenizer_gate import FrozenBatchTokenizer
        tokenizer = FrozenBatchTokenizer(tokenizer_path)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result = convert(records, tokenizer, external_path(args.output, fresh=True))
    result.update(tokenizer_sha256=args.tokenizer_sha256, encoder_kind=args.encoder_kind,
                  source_records_sha256=file_sha256(input_path))
    (external_path(args.output) / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
