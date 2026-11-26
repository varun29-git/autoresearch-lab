from io import BytesIO
import hashlib
import json
import unittest

import numpy as np

from production.packed_index import descriptor_offsets


class PackedIndexTests(unittest.TestCase):
    def test_offsets_preserve_unicode_descriptors_across_arbitrary_chunks(self):
        rows=[json.dumps({'text':text},ensure_ascii=False).encode()+b'\n'
              for text in ['Hindi हिन्दी','café','line\ninside','🚀']]
        raw=b''.join(rows);out=BytesIO()
        descriptor_offsets([raw[i:i+3]for i in range(0,len(raw),3)],out,
            expected_bytes=len(raw),expected_sha256=hashlib.sha256(raw).hexdigest(),sequences=4)
        offsets=np.frombuffer(out.getvalue(),dtype='<u8')
        self.assertEqual(list(offsets),[0]+list(np.cumsum(list(map(len,rows)))))
        for i,row in enumerate(rows):self.assertEqual(raw[offsets[i]:offsets[i+1]],row)

    def test_corrupt_or_truncated_sidecar_cannot_be_accepted(self):
        raw=b'{"segments":[]}\n'
        for changed,count in [(raw[:-1],1),(raw+b'\n',1),(raw,2)]:
            with self.assertRaises(ValueError):
                descriptor_offsets([changed],BytesIO(),expected_bytes=len(raw),
                    expected_sha256=hashlib.sha256(raw).hexdigest(),sequences=count)


if __name__=='__main__':unittest.main()
