"""Ordered native encoding across independent per-thread tokenizer caches."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading

from .tokenizer_native import Native65kTokenizer


class NativeBatchPool:
    def __init__(self, path, digest, workers):
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("native encoding thread count must be between 1 and 32")
        self.workers = workers; self.local = threading.local()
        def initialize():
            self.local.tokenizer = Native65kTokenizer(path, expected_sha256=digest)
        self.pool = ThreadPoolExecutor(max_workers=workers, initializer=initialize)

    def _encode(self, chunk):
        return self.local.tokenizer.encode_batch_bytes(chunk)

    def encode_batch_bytes(self, texts):
        if not texts: return []
        count = min(self.workers, len(texts))
        # Spread variable-length documents across lanes; restore exact source
        # order after the C++ work, without materializing Python token integers.
        chunks = [texts[lane::count] for lane in range(count)]
        output = [None] * len(texts)
        for lane, values in enumerate(self.pool.map(self._encode, chunks)):
            output[lane::count] = values
        return output

    def close(self): self.pool.shutdown(wait=True)


@lru_cache(maxsize=1)
def native_batch_pool(path, digest, workers):
    return NativeBatchPool(path, digest, workers)
