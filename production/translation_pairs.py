"""Qualified, unlabeled translation pairs in one causal attention segment.

This CPU constructor is deliberately independent of the paid preparation DAG.
It consumes whole source records, not matching question IDs or decoded bins.
Production promotion additionally needs global ownership/holdout reconciliation
and a segment-aware native loader. Source evidence stays outside Git.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata

import numpy as np

from production.document_packing import Fragment
from production.packing_worker import fingerprint
from terminus.tokenizer65k import ATOM_REVISION_SHA256


def text_sha(text):
    if not isinstance(text, str) or not text:
        raise ValueError("missing original Unicode text")
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def number_signature(text):
    # Native decimal scripts are equivalent; ambiguous punctuation or changed
    # quantities conservatively reject a candidate rather than rewriting it.
    text = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    return Counter(re.findall(r"[0-9]+(?:[.,][0-9]+)*", text))


@dataclass(frozen=True)
class Passage:
    record: dict
    metadata: dict
    language: str
    leaf: str
    stratum: str
    original: bool = False


def lineage(passage, exclusions, minimum_score=95):
    r, m = passage.record, passage.metadata
    if exclusions.excludes(r):
        raise ValueError("translation family or document is held out")
    if not passage.language or not passage.leaf or not passage.stratum:
        raise ValueError("missing language/source quota identity")
    text = m.get('text')
    if text_sha(text) != r['source_text_sha256']:
        raise ValueError("raw passage differs from the frozen conversion index")
    if passage.original:
        seed, uid = text, m.get('uuid')
    else:
        seed, uid = m.get('source_text'), m.get('source_uuid')
        text_sha(seed)
        try:
            score = float(m['overall_score'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("missing translation quality score") from exc
        evaluation = m.get('evaluation_result')
        if (not math.isfinite(score) or not minimum_score <= score <= 100
                or not isinstance(evaluation, dict)
                or not isinstance(evaluation.get('error_analysis'), list)):
            raise ValueError("translation quality evidence is insufficient")
        for error in evaluation['error_analysis']:
            if not isinstance(error, dict) or str(error.get('severity', '')).lower() != 'minor':
                raise ValueError("translation has major, critical or unclassified errors")
        if number_signature(seed) != number_signature(text):
            raise ValueError("translation changes numeric quantities")
        # Fenced code/formula blocks are immutable in this conservative subset.
        # Translated code comments can be qualified separately using AST checks.
        blocks = lambda s: Counter(re.findall(r"```[^\n]*\n(.*?)```|\$\$(.*?)\$\$", s, re.S))
        if blocks(seed) != blocks(text):
            raise ValueError("translation changes protected code or formula blocks")
    identity = (m.get('_source_repo'), m.get('_source_config'),
                m.get('_source_file'), str(uid) if uid else '', text_sha(seed))
    if any(not value for value in identity):
        raise ValueError("missing original source lineage")
    return identity


def compound_pair(left, right, left_tokens, right_tokens, *, tokenizer,
                  exclusions, length=4096, minimum_score=95):
    """Check source and encoder evidence, then emit exactly one fragment.

    Every eligible language pair is supported. No arbitrary truncation or
    language tags are introduced. Two independently tokenized passages must
    agree exactly with encoding their blank-line-joined original texts.
    """
    if tokenizer.sha256 != ATOM_REVISION_SHA256:
        raise ValueError("pair encoder differs from the production tokenizer")
    if left.language == right.language or left.original and right.original:
        raise ValueError("pair needs two languages and translation evidence")
    a = lineage(left, exclusions, minimum_score)
    b = lineage(right, exclusions, minimum_score)
    if a != b:
        raise ValueError("passages do not translate the same original document")
    family = fingerprint(['translation_lineage', *a])
    # Also check the original UUID/hash, even if a published per-language
    # split_family_id accidentally differs between counterpart rows.
    if exclusions.excludes({'doc_id': a[3], 'row_key': family,
            'split_family_id': a[3], 'validation': False}):
        raise ValueError("original translation family is held out")
    for p, tokens in ((left, left_tokens), (right, right_tokens)):
        values = np.asarray(tokens)
        if (values.ndim != 1 or values.dtype.kind not in 'ui' or not len(values)
                or np.any(values < 32) or np.any(values >= 65536)
                or len(values) != p.record['n_tokens']
                or not np.array_equal(values, tokenizer.encode(p.metadata['text']))):
            raise ValueError("pair tokens differ from the whole frozen source record")
    if left.record['source_text_sha256'] == right.record['source_text_sha256']:
        raise ValueError("identical source text is not a cross-language pair")
    separator = tokenizer.encode('\n\n')
    joined = np.concatenate((left_tokens, separator, right_tokens)).astype('<u2')
    if not np.array_equal(joined, tokenizer.encode(left.metadata['text']+'\n\n'+right.metadata['text'])):
        raise ValueError("joined encoding changes a source boundary")
    if len(joined) > length:
        raise ValueError("whole translation pair exceeds context; use an ordinary document")
    eod_input = len(joined) < length
    if eod_input:
        joined = np.concatenate((joined, np.array([4], dtype='<u2')))
    components = []
    start = 0
    for p, tokens in ((left, left_tokens), (right, right_tokens)):
        components.append({'start': start, 'end': start+len(tokens),
            'language': p.language, 'leaf': p.leaf, 'stratum': p.stratum,
            'content_tokens': len(tokens), 'document': p.record})
        start += len(tokens)+len(separator)
    document = {'kind': 'unlabeled_translation_pair', 'family_id': family,
        'lineage': list(a), 'components': components,
        'separator_start': len(left_tokens), 'separator_tokens': list(separator),
        'tokenizer_sha256': tokenizer.sha256, 'quality_minimum_score': minimum_score}
    return Fragment(joined, document, 0, True, None if eod_input else 4,
                    eod_input, len(separator))


def validate_components(tokens, document, *, eod_input, content_tokens, synthetic_tokens):
    """Verify compound source spans and inserted separator against the payload."""
    components=document.get('components', [])
    separator=document.get('separator_tokens', [])
    if (len(components)!=2 or not separator or len(separator)!=synthetic_tokens
            or document.get('tokenizer_sha256')!=ATOM_REVISION_SHA256):
        raise ValueError('invalid compound source or separator metadata')
    a,b=components
    if (a['start']!=0 or a['end']!=document['separator_start']
            or b['start']!=a['end']+len(separator)
            or b['end']!=len(tokens)-int(eod_input)
            or a['language']==b['language']
            or a['document']['source_text_sha256']==b['document']['source_text_sha256']
            or not np.array_equal(tokens[a['end']:b['start']],separator)):
        raise ValueError('compound spans differ from the token payload')
    for component in components:
        if (type(component['content_tokens'])is not int or component['content_tokens']<=0
                or component['end']-component['start']!=component['content_tokens']
                or component['document']['n_tokens']!=component['content_tokens']):
            raise ValueError('compound source counters differ')
    if sum(c['content_tokens']for c in components)!=content_tokens:
        raise ValueError('compound content differs from its source quotas')


class PairBudget:
    """Single-coordinator accounting; pair records replace ordinary occurrences.

    This is not a distributed ownership ledger. Before using it with workers,
    bind their selections and all protected hashes to one reconciled commit.
    Both sides count toward their existing language/source/child quotas.
    """
    def __init__(self, remaining, *, ceiling=20_000_000_000, protected_hashes=()):
        if type(ceiling) is not int or ceiling < 0:
            raise ValueError("invalid paired-content ceiling")
        self.remaining = dict(remaining)
        if any(type(v) is not int or v < 0 for v in self.remaining.values()):
            raise ValueError("invalid source/stratum quota")
        self.ceiling = ceiling
        self.content_tokens = 0
        self.used = set(protected_hashes)
        self.families = set()

    def claim(self, pair, *, phase):
        if phase != 'dense-stable' or pair.document.get('kind') != 'unlabeled_translation_pair':
            raise ValueError("pairs are restricted to ordinary dense stable data")
        validate_components(pair.tokens,pair.document,eod_input=pair.eod_input,
            content_tokens=pair.content_tokens,synthetic_tokens=pair.synthetic_tokens)
        components = pair.document['components']
        hashes = {p['document']['source_text_sha256'] for p in components}
        if (len(hashes) != 2 or hashes & self.used
                or pair.document['family_id'] in self.families):
            raise ValueError("pair would reuse a protected record or translation family")
        charges = Counter()
        for p in components:
            charges[(p['leaf'], p['stratum'])] += p['content_tokens']
        if sum(charges.values()) != pair.content_tokens:
            raise ValueError("pair component token counters differ")
        if self.content_tokens+pair.content_tokens > self.ceiling:
            return False
        if any(n > self.remaining.get(k, 0) for k, n in charges.items()):
            return False
        # All checks precede mutation: a rejected pair consumes nothing.
        for k, n in charges.items():
            self.remaining[k] -= n
        self.content_tokens += pair.content_tokens
        self.used.update(hashes)
        self.families.add(pair.document['family_id'])
        return True
