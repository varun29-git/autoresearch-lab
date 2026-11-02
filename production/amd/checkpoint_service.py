"""Bounded active uploads, verified local retention and durable restore."""
import argparse
import json
import math
from pathlib import Path
import queue
import re
import shutil
import threading
import time

from production.artifacts import external_path
from terminus.checkpoint import checkpoint_lock, load_checkpoint
from terminus.integrity import file_sha256, fingerprint, fsync_directory, write_json_atomic
from .checkpoint_store import LocalStore, make_store


def step_name(name):
    if not isinstance(name, str) or re.fullmatch(r'step_[0-9]{8,}\.pt', name) is None:
        raise ValueError('invalid checkpoint step name')
    return name


def marker_name(name):
    return step_name(name)[:-3] + '.durable.json'


def verify_durable(store, receipt):
    name = step_name(receipt['file'])
    marker = marker_name(name)
    if not store.exists(marker):
        return False
    record = store.read_json(marker)
    return (record.get('receipt') == receipt and record.get('schema') == 1
            and store.verify(name, receipt['sha256'], receipt['size_bytes'])
            and store.read_json(name[:-3] + '.json') == receipt)


class CheckpointService:
    def __init__(self, run_dir, store, *, keep_last=3, protected=(), max_pending_bytes=32*1024**3,
                 max_pending_seconds=1800., min_free_bytes=32*1024**3, timeout=600.):
        if type(keep_last) is not int or keep_last < 2:
            raise ValueError('retain at least two local checkpoints')
        if (type(max_pending_bytes) is not int or max_pending_bytes <= 0
                or type(min_free_bytes) is not int or min_free_bytes < 0
                or any(not math.isfinite(x) or x <= 0 for x in (max_pending_seconds, timeout))):
            raise ValueError('invalid backup resource limits')
        self.run = external_path(run_dir); self.root = self.run / 'checkpoints'
        self.root.mkdir(exist_ok=True)
        self.receipts = self.root / '.durable'; self.receipts.mkdir(exist_ok=True)
        if isinstance(store, LocalStore) and (store.root == self.run or self.run in store.root.parents or store.root in self.run.parents):
            raise ValueError('durable destination must be separate from the run directory')
        self.store = store
        self.keep_last, self.protected = keep_last, {step_name(n) for n in protected}
        self.max_bytes, self.max_age, self.min_free, self.timeout = max_pending_bytes, max_pending_seconds, min_free_bytes, timeout
        self.pending = {}; self.failure = None; self.closed = False
        self.mutex = threading.Lock(); self.tasks = queue.Queue()
        self.run_sha = file_sha256(self.run / 'run.json')
        store.put(self.run / 'run.json', 'run.json', self.run_sha)
        self.worker = threading.Thread(target=self._work, daemon=True, name='ci1-durable-checkpoints')
        self.worker.start()

    def _local_marker(self, name, receipt):
        path = self.receipts / marker_name(name)
        if not path.exists():
            return False
        record = json.loads(path.read_text())
        return record.get('receipt') == receipt and record.get('destination') == self.store.identity

    def enqueue(self, path):
        name = step_name(Path(path).name)
        receipt_path = self.root / (name[:-3] + '.json')
        receipt = json.loads(receipt_path.read_text())
        if receipt['file'] != name:
            raise ValueError('checkpoint receipt name mismatch')
        if self._local_marker(name, receipt):
            return
        with self.mutex:
            if self.closed or self.failure:
                raise RuntimeError('checkpoint upload service stopped or failed')
            if name in self.pending:
                return
            size = receipt['size_bytes']
            if sum(item['size'] for item in self.pending.values()) + size > self.max_bytes:
                raise RuntimeError('checkpoint upload backlog exceeds byte limit')
            self.pending[name] = {'size':size, 'since':(self.root/name).stat().st_mtime}
            self.tasks.put(name)

    def _upload(self, name):
        path = self.root / name; receipt_path = path.with_suffix('.json')
        receipt = json.loads(receipt_path.read_text())
        if receipt['size_bytes'] != path.stat().st_size or receipt['sha256'] != file_sha256(path):
            raise ValueError('local checkpoint failed verification before upload')
        self.store.put(path, name, receipt['sha256'])
        self.store.put(receipt_path, receipt_path.name, file_sha256(receipt_path))
        record = {'schema':1, 'receipt':receipt, 'run_sha256':self.run_sha, 'destination':self.store.identity}
        candidate = self.receipts / (marker_name(name) + '.pending')
        write_json_atomic(candidate, record)
        self.store.put(candidate, marker_name(name), file_sha256(candidate))
        if not verify_durable(self.store, receipt):
            raise ValueError('remote checkpoint/receipt/marker verification failed')
        # The local durable marker exists ONLY after the remote commit succeeds.
        write_json_atomic(self.receipts / marker_name(name), record)
        candidate.unlink(missing_ok=True)

    def _work(self):
        while True:
            name = self.tasks.get()
            if name is None:
                self.tasks.task_done(); return
            try:
                self._upload(name)
                self.prune()
            except Exception as exc:
                with self.mutex:
                    self.failure = {'checkpoint':name, 'error_type':type(exc).__name__, 'error':str(exc)}
                self.tasks.task_done(); return
            with self.mutex:
                self.pending.pop(name)
            self.tasks.task_done()

    def health(self):
        with self.mutex:
            pending = dict(self.pending); failure = self.failure
        age = max((time.time()-item['since'] for item in pending.values()), default=0.)
        free = shutil.disk_usage(self.run).free
        result = {'schema':1, 'pending_checkpoints':list(pending), 'pending_bytes':sum(x['size'] for x in pending.values()),
                  'oldest_pending_seconds':age, 'free_bytes':free, 'failure':failure, 'time_unix':time.time()}
        write_json_atomic(self.run / 'backup_health.json', result)
        if failure:
            raise RuntimeError(f'checkpoint backup failed: {failure}')
        if age > self.max_age or free < self.min_free:
            raise RuntimeError('checkpoint backup age or free-space limit exceeded')
        return result

    def scan(self):
        # Producer only publishes a receipt after an immutable payload exists.
        with checkpoint_lock(self.root):
            for receipt_path in sorted(self.root.glob('step_*.json')):
                if re.fullmatch(r'step_[0-9]{8,}\.json', receipt_path.name):
                    self.enqueue(receipt_path.with_suffix('.pt'))
        self.health()
        # Remote verification/retention belongs to the upload worker, never
        # the rank-zero training poll. close() performs one final prune.

    def prune(self):
        receipts = [json.loads(p.read_text()) for p in sorted(self.root.glob('step_*.json'))
                    if re.fullmatch(r'step_[0-9]{8,}\.json', p.name)]
        receipts.sort(key=lambda r:r['state']['updates'])
        keep = self.protected | {r['file'] for r in receipts[-self.keep_last:]}
        removed = []
        for receipt in receipts:
            name = receipt['file']
            if name in keep or not self._local_marker(name, receipt):
                continue
            if not verify_durable(self.store, receipt):
                raise RuntimeError('durable copy changed; refusing local retention')
            with checkpoint_lock(self.root, exclusive=True):
                latest = json.loads((self.root/'latest.json').read_text())
                if name == latest['file']:
                    continue
                path = self.root / step_name(name)
                if json.loads(path.with_suffix('.json').read_text()) != receipt:
                    raise RuntimeError('receipt changed during retention')
                # A reader holding the shared lock has already loaded/mapped its payload.
                path.with_suffix('.json').unlink(); path.unlink()
                fsync_directory(self.root)
                removed.append(name)
        return removed

    def close(self):
        with self.mutex:
            if not self.closed:
                self.closed = True; self.tasks.put(None)
        self.worker.join(timeout=self.timeout)
        if self.worker.is_alive():
            raise TimeoutError('checkpoint upload queue did not drain before deadline')
        self.health()
        with self.mutex:
            if self.pending:
                raise RuntimeError('checkpoint uploads remain incomplete')
        self.prune()


def restore(store, name, output):
    name = step_name(name); output = external_path(output, fresh=True)
    marker = store.read_json(marker_name(name)); receipt = marker['receipt']
    if receipt['file'] != name or not verify_durable(store, receipt):
        raise ValueError('checkpoint is not durably committed')
    output.mkdir(parents=True)
    root = output / 'checkpoints'; root.mkdir()
    store.download('run.json', output/'run.json')
    if file_sha256(output/'run.json') != marker['run_sha256']:
        raise ValueError('restored run identity mismatch')
    run = json.loads((output/'run.json').read_text())
    if fingerprint(run['contract']) != receipt['contract_sha256'] or run['contract_sha256'] != receipt['contract_sha256']:
        raise ValueError('restore contract mismatch')
    store.download(name, root/name)
    write_json_atomic(root/(name[:-3]+'.json'), receipt)
    load_checkpoint(root/name, run['contract'])  # Hash, size, contract and counters before publishing latest.
    write_json_atomic(root/'latest.json', receipt)
    return {'status':'restored_verified', 'checkpoint':name, 'output_dir':str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    print(json.dumps(restore(make_store(args.destination), args.checkpoint, args.output_dir)))


if __name__ == '__main__':
    main()
