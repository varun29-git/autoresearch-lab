import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from production.document_packing import ShardWriter, causal_mask, decode_pack, encode_pack, fragments
from production.translation_pairs import Passage, PairBudget, compound_pair, text_sha
from production.packing_parallel import pack_chunk
from terminus.tokenizer65k import ATOM_REVISION_SHA256


class Encoder:
    sha256 = ATOM_REVISION_SHA256

    def encode(self, text):
        return [ord(c)+32 for c in text]


class Exclusions:
    def __init__(self, families=()):
        self.families = set(families)

    def excludes(self, record):
        return record.get('validation') is not False or record['split_family_id'] in self.families


def passage(text, language, *, original=False, seed='The answer is 42.', uid='answer-variant-a'):
    m = {'text': text, 'source_text': seed, 'source_uuid': uid,
        'uuid': uid if original else language+'-'+uid, '_source_repo': 'source/repo',
        '_source_config': 'config', '_source_file': 'file.parquet',
        'overall_score': '98', 'evaluation_result': {'error_analysis': []}}
    r = {'source_text_sha256': text_sha(text), 'doc_id': m['uuid'],
         'row_key': language+':123', 'split_family_id': uid,
         'validation': False, 'n_tokens': len(Encoder().encode(text))}
    return Passage(r, m, language, language+'/source', 'domain-child', original)


def pair(a=None, b=None, **kwargs):
    a = a or passage('The answer is 42.', 'en', original=True)
    b = b or passage('उत्तर ४२ है।', 'hi')
    return compound_pair(a, b, Encoder().encode(a.metadata['text']),
        Encoder().encode(b.metadata['text']), tokenizer=Encoder(),
        exclusions=Exclusions(), **kwargs)


class TranslationPairTests(unittest.TestCase):
    def test_unlabeled_pair_shares_global_segment_and_blocks_unrelated_doc(self):
        p = pair()
        other = next(fragments(np.array([100,101]), {'id': 'unrelated'}))
        x, d = encode_pack([p, other], length=128)
        y, positions, segments = decode_pack(x, d)
        second = p.document['components'][1]['start']
        self.assertEqual(len(d['segments']), 2)
        self.assertEqual(positions[second], second)
        self.assertEqual(segments[second], 0)
        self.assertEqual(y[second-1], x[second])
        mask = causal_mask(segments, positions)
        self.assertTrue(mask[second, 0])
        self.assertFalse(mask[len(p.tokens), :len(p.tokens)].any())
        self.assertEqual(d['content_tokens'], p.content_tokens+2)
        self.assertEqual(d['synthetic_tokens'], 2)
        self.assertEqual(d['used_slots'], d['content_tokens']+2+2)

    def test_any_language_pair_and_local_window(self):
        a = passage('अ'*600+' 42', 'mr')
        b = passage('ఆ'*600+' 42', 'te')
        p = pair(a,b)
        x,d=encode_pack([p]);_,positions,segments=decode_pack(x,d)
        second=p.document['components'][1]['start']
        mask=causal_mask(segments[:len(p.tokens)],positions[:len(p.tokens)],window=512)
        self.assertTrue(mask[second,second-1])
        self.assertFalse(mask[second,0])
        self.assertTrue(causal_mask(segments[:len(p.tokens)],positions[:len(p.tokens)])[second,0])

    def test_full_context_uses_only_final_eod_target(self):
        length=len('The answer is 42.\n\nउत्तर ४२ है।')
        p=pair(length=length)
        self.assertFalse(p.eod_input)
        x,d=encode_pack([p],length=length);y,_,_=decode_pack(x,d)
        self.assertEqual(y[-1],4)
        self.assertNotIn(4,x)
        with self.assertRaisesRegex(ValueError,'exceeds context'):
            pair(length=length-1)

    def test_reject_shared_id_with_changed_answer_and_unqualified_translation(self):
        for mutation in (
            lambda m:m.update(source_text='Different answer 42.'),
            lambda m:m.update(source_uuid='different-variant'),
            lambda m:m.update(overall_score=88),
            lambda m:m.update(overall_score='nan'),
            lambda m:m.update(evaluation_result={'error_analysis':[{'severity':'Major'}]}),
            lambda m:m.pop('_source_file'),
        ):
            b=passage('उत्तर ४२ है।','hi');mutation(b.metadata)
            with self.assertRaises(ValueError):pair(b=b)
        with self.assertRaisesRegex(ValueError,'numeric'):
            pair(b=passage('उत्तर ४३ है।','hi'))

    def test_heldout_original_family_blocks_different_translation_split_id(self):
        a=passage('उत्तर ४२ है।','hi');b=passage('సమాధానం 42','te')
        a.record['split_family_id']='translation-local-a';b.record['split_family_id']='translation-local-b'
        with self.assertRaisesRegex(ValueError,'original translation family'):
            compound_pair(a,b,Encoder().encode(a.metadata['text']),Encoder().encode(b.metadata['text']),
                tokenizer=Encoder(),exclusions=Exclusions(['answer-variant-a']))

    def test_token_stream_mismatch_and_boundary_change_rejected(self):
        a=passage('The answer is 42.','en',original=True);b=passage('उत्तर ४२ है।','hi')
        with self.assertRaisesRegex(ValueError,'frozen source'):
            compound_pair(a,b,[100]*a.record['n_tokens'],Encoder().encode(b.metadata['text']),
                tokenizer=Encoder(),exclusions=Exclusions())
        class BadJoin(Encoder):
            def encode(self,text):
                return super().encode(text)+([100] if '\n\n' in text and text!='\n\n' else [])
        with self.assertRaisesRegex(ValueError,'joined encoding'):
            compound_pair(a,b,Encoder().encode(a.metadata['text']),Encoder().encode(b.metadata['text']),
                tokenizer=BadJoin(),exclusions=Exclusions())

    def test_quota_claim_is_atomic_and_blocks_replay_protected_and_non_dense(self):
        p=pair();quota={(c['leaf'],c['stratum']):c['content_tokens']for c in p.document['components']}
        budget=PairBudget(quota,ceiling=p.content_tokens-1)
        self.assertFalse(budget.claim(p,phase='dense-stable'))
        self.assertEqual(budget.remaining,quota);self.assertEqual(budget.used,set())
        budget.ceiling=p.content_tokens
        self.assertTrue(budget.claim(p,phase='dense-stable'))
        self.assertFalse(any(budget.remaining.values()))
        with self.assertRaises(ValueError):budget.claim(p,phase='dense-stable')
        with self.assertRaises(ValueError):PairBudget(quota).claim(p,phase='qat-stable')
        with self.assertRaises(ValueError):
            PairBudget(quota,protected_hashes=[next(iter(budget.used))]).claim(p,phase='dense-stable')

    def test_inserted_tokens_propagate_to_writer_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=ShardWriter(tmp,'pair',length=128,audit_stride=1)
            writer.add([pair()]);r=writer.close()
            self.assertEqual(r['synthetic_tokens'],2)
        x,d=encode_pack([pair()],length=128);d['segments'][0]['synthetic_tokens']=1
        with self.assertRaises(ValueError):decode_pack(x,d)
        x,d=encode_pack([pair()],length=128)
        d['segments'][0]['document']['components'][1]['content_tokens']-=1
        with self.assertRaises(ValueError):decode_pack(x,d)
        x,d=encode_pack(list(fragments(np.array([101,102]),{})),length=8)
        self.assertNotIn('synthetic_tokens',d)
        self.assertNotIn('synthetic_tokens',d['segments'][0])

    def test_encoded_chunks_preserve_inserted_token_counters(self):
        with tempfile.TemporaryDirectory()as tmp:
            chunk=pack_chunk([pair()],128,3,tmp)
            self.assertEqual(chunk['names'][-1],'synthetic_tokens')
            self.assertEqual(int(chunk['prefixes'][-1,-1]),2)
            writer=ShardWriter(tmp,'assembled',length=128)
            counts={'sequences':chunk['sequences']}|dict(zip(chunk['names'],map(int,chunk['prefixes'][-1])))
            writer.append_encoded(Path(chunk['payload']['path']).read_bytes(),Path(chunk['index']['path']).read_bytes(),counts)
            self.assertEqual(writer.close()['synthetic_tokens'],2)


if __name__=='__main__':unittest.main()
