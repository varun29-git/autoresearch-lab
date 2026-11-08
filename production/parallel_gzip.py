"""Bounded ordered gzip members compressed on independent native threads."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gzip


class ParallelGzipWriter:
    def __init__(self,path,*,workers=4,block_bytes=4<<20):
        if workers<1 or block_bytes<1:raise ValueError('invalid gzip pipeline bounds')
        self.stream=open(path,'wb',buffering=8<<20)
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.limit=workers*2;self.block_bytes=block_bytes
        self.buffer=bytearray();self.pending=deque();self.closed=False

    def write(self,data):
        if self.closed:raise ValueError('write to closed gzip stream')
        self.buffer.extend(data)
        if len(self.buffer)>=self.block_bytes:self.flush_block()
        return len(data)

    def flush_block(self):
        if self.buffer:
            self.pending.append(self.pool.submit(gzip.compress,bytes(self.buffer),compresslevel=1,mtime=0))
            self.buffer.clear()
        if len(self.pending)>=self.limit:self.stream.write(self.pending.popleft().result())

    def close(self):
        if self.closed:return
        try:
            self.flush_block()
            if not self.pending and self.stream.tell()==0:self.stream.write(gzip.compress(b'',mtime=0))
            while self.pending:self.stream.write(self.pending.popleft().result())
        finally:
            self.pool.shutdown(wait=True,cancel_futures=True)
            self.stream.close();self.closed=True
