import tempfile
import unittest
from pathlib import Path

import numpy as np

from production.document_packing import (ShardWriter, best_fit, causal_mask,
    decode_pack, encode_pack, file_sha, fragments)


class DocumentPackingTests(unittest.TestCase):
    def test_every_intended_target_once_across_long_boundaries(self):
        for size in (1, 7, 8, 9, 15, 16, 17, 31, 511, 512, 513, 4095, 4096, 4097):
            with self.subTest(size=size):
                length = 8 if size < 32 else 4096
                source = np.arange(100, size + 100, dtype=np.uint16)
                fs = list(fragments(source, {"id": "a"}, length=length))
                observed = []
                recovered = []
                for f in fs:
                    x, d = encode_pack([f], length=length)
                    y, _, _ = decode_pack(x, d)
                    observed.extend(y[y >= 0].tolist())
                    recovered.extend(f.tokens[:f.content_tokens].tolist())
                self.assertEqual(recovered, source.tolist())
                self.assertEqual(observed, source[1:].tolist() + [4])
                self.assertEqual(sum(f.document_end for f in fs), 1)

    def test_short_documents_do_not_predict_each_other(self):
        fs = [next(fragments(np.array(x), {"id": str(i)}, length=8))
              for i, x in enumerate(([101, 102], [201, 202, 203]))]
        x, d = encode_pack(fs, length=8)
        y, positions, seg = decode_pack(x, d)
        self.assertEqual(x.tolist(), [101, 102, 4, 201, 202, 203, 4, 3])
        self.assertEqual(y.tolist(), [102, 4, -100, 202, 203, 4, -100, -100])
        mask = causal_mask(seg, positions)
        self.assertFalse(mask[3:, :3].any())
        self.assertFalse(mask[7].any())
        self.assertTrue(mask[6, 3:7].all())

    def test_local_window_boundary_and_global_prefix(self):
        source = np.full(1200, 100, dtype=np.uint16)
        x, d = encode_pack(list(fragments(source, {"id": "a"})), length=4096)
        _, positions, seg = decode_pack(x, d)
        local = causal_mask(seg[:1200], positions[:1200], window=512)
        glob = causal_mask(seg[:1200], positions[:1200])
        self.assertEqual(int(local[1000].sum()), 512)
        self.assertTrue(local[1000, 489])
        self.assertFalse(local[1000, 488])
        self.assertTrue(glob[1000, 0])
        self.assertFalse(glob[1000, 1001])

    def test_quota_boundary_is_not_document_end_or_future_target(self):
        source = np.arange(100, 117, dtype=np.uint16)
        fs = list(fragments(source, {"id": "a"}, length=8, document_end=False))
        labels = []
        for f in fs:
            x, d = encode_pack([f], length=8)
            y, _, _ = decode_pack(x, d)
            labels.extend(y[y >= 0].tolist())
        self.assertEqual(labels, source[1:].tolist())
        self.assertFalse(any(f.document_end or f.eod_input for f in fs))

    def test_bfd_preserves_all_fragments_and_packs_tightly(self):
        fs = [next(fragments(np.full(n - 1, i + 100, dtype=np.uint16),
                            {"id": str(i)}, length=16))
              for i, n in enumerate([10, 6, 9, 7, 8, 8])]
        packs = best_fit(fs, length=16, seed=18)
        self.assertEqual(len(packs), 3)
        self.assertEqual(sorted(id(x) for p in packs for x in p), sorted(map(id, fs)))
        self.assertTrue(all(sum(len(x.tokens) for x in p) == 16 for p in packs))
        self.assertEqual([[x.document for x in p] for p in packs],
            [[x.document for x in p] for p in best_fit(fs, length=16, seed=18)])

    def test_randomized_packing_conserves_content_and_targets(self):
        rng = np.random.default_rng(91)
        fs = []
        expected = 0
        for i, n in enumerate(rng.integers(1, 1000, size=400)):
            expected += int(n)
            fs.extend(fragments(np.full(n, 100 + i, dtype=np.uint16),
                                {"id": str(i)}, length=128))
        counts = [0, 0]
        for pack in best_fit(fs, length=128, seed=3):
            x, d = encode_pack(pack, length=128)
            y, _, _ = decode_pack(x, d)
            counts[0] += d['content_tokens']
            counts[1] += int((y >= 0).sum())
        self.assertEqual(counts, [expected, expected])

    def test_writer_payload_descriptor_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            w = ShardWriter(temp, 'shard', length=8, audit_stride=1)
            fs = list(fragments(np.arange(100, 121, dtype=np.uint16),
                                {"id": "source-row"}, length=8))
            for p in best_fit(fs, length=8):
                w.add(p)
            receipt = w.close()
            self.assertEqual(receipt['content_tokens'], 21)
            self.assertEqual(receipt['valid_targets'], 21)
            self.assertEqual(receipt['payload']['bytes'], 3 * 8 * 2)
            self.assertEqual(file_sha(receipt['payload']['path']), receipt['payload']['sha256'])
            self.assertFalse(receipt['native_training_accepted'])
            self.assertEqual(len(Path(receipt['index']['path']).read_text().splitlines()), 3)

    def test_reject_control_tokens_and_corrupt_descriptors(self):
        for token in (3, 4, 5, 31, 65536, -1):
            with self.assertRaises(ValueError):
                list(fragments(np.array([100, token]), {}))
        x, d = encode_pack(list(fragments(np.array([100, 101]), {})), length=8)
        d['valid_targets'] += 1
        with self.assertRaises(ValueError):
            decode_pack(x, d)


if __name__ == '__main__':
    unittest.main()
