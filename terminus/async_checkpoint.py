"""One bounded immutable CPU snapshot; background serialization and hashing."""
from concurrent.futures import Future
import math
import threading

from .checkpoint import checkpoint_payload, commit_checkpoint_payload, freeze_payload


class AsyncCheckpointWriter:
    def __init__(self, directory, *, timeout=600., after_commit=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("checkpoint timeout must be finite and positive")
        self.directory, self.timeout = directory, timeout
        self.after_commit = after_commit
        self.future = None
        self.thread = None
        self.closed = False

    def check(self):
        if self.future is not None and self.future.done():
            self.future.result()  # Background failures must reach the caller.

    def flush(self):
        if self.future is not None:
            self.future.result(timeout=self.timeout)
            self.thread.join(timeout=self.timeout)
            self.future = self.thread = None

    def submit(self, model, optimizer, state, contract, rank_rng):
        if self.closed:
            raise RuntimeError("checkpoint writer is closed")
        self.flush()  # Backpressure BEFORE allocating a second full snapshot.
        snapshot = freeze_payload(checkpoint_payload(model, optimizer, state, contract, rank_rng))
        future = Future()
        self.future = future

        def write():
            try:
                path = commit_checkpoint_payload(self.directory, snapshot)
                if self.after_commit is not None:
                    self.after_commit(path)
                future.set_result(path)
            except BaseException as exc:
                future.set_exception(exc)

        self.thread = threading.Thread(target=write, daemon=True, name="ci1-checkpoint")
        self.thread.start()

    def close(self):
        self.closed = True
        self.flush()
