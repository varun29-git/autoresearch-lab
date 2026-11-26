"""Opt-in production operations integrated at complete-update boundaries."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch.distributed as dist

from production.artifacts import external_path
from terminus.integrity import fingerprint, write_json_atomic
from .allocation import load as load_allocation, remaining
from .checkpoint_service import CheckpointService, verify_durable
from .checkpoint_store import make_store


@dataclass(frozen=True)
class OperationsConfig:
    allocation_record: str
    backup_destination: str
    schema: int = 1
    shutdown_reserve_seconds: float = 1800.
    keep_last: int = 3
    protected: tuple = ()
    max_pending_bytes: int = 32*1024**3
    max_pending_seconds: float = 1800.
    min_free_bytes: int = 32*1024**3
    io_timeout_seconds: float = 600.
    health_interval_updates: int = 10

    def __post_init__(self):
        import math
        if self.schema!=1 or not self.backup_destination or type(self.health_interval_updates) is not int or self.health_interval_updates<=0:
            raise ValueError('invalid operations schema/destination/cadence')
        if (not math.isfinite(self.shutdown_reserve_seconds) or self.shutdown_reserve_seconds<=0
                or not math.isfinite(self.io_timeout_seconds) or self.io_timeout_seconds<=0
                or self.shutdown_reserve_seconds<2*self.io_timeout_seconds):
            raise ValueError('shutdown reserve must fund checkpoint and upload timeouts')
        external_path(self.allocation_record)


def load_config(path):
    return OperationsConfig(**json.loads(external_path(path).read_text()))


class TrainingOperations:
    def __init__(self, workload, config):
        self.workload, self.config = workload, config
        self.allocation=load_allocation(config.allocation_record)
        if self.allocation['gpus']!=workload.world_size:
            raise ValueError('allocation GPU count does not match the entire training node')
        self.service=None; self.closed=False
        def setup():
            budget=remaining(self.allocation,reserve_seconds=config.shutdown_reserve_seconds)
            if budget['training_seconds']<=0:raise RuntimeError('allocation has no training time left')
            root=workload.output_dir
            identity=json.loads(json.dumps({'operations':asdict(config),'allocation':self.allocation['identity_sha256']}))
            path=root/'operations.json'
            if path.exists() and json.loads(path.read_text())!=identity:
                raise ValueError('operations/allocation changed across resume')
            write_json_atomic(path,identity)
            self.service=CheckpointService(root,make_store(config.backup_destination),keep_last=config.keep_last,
                protected=config.protected,max_pending_bytes=config.max_pending_bytes,
                max_pending_seconds=config.max_pending_seconds,min_free_bytes=config.min_free_bytes,
                timeout=config.io_timeout_seconds)
            if workload.checkpoint_writer is not None:
                workload.checkpoint_writer.after_commit=self.service.enqueue
            self.service.scan()
        try:
            self._collective(setup)
            self.poll(force=True)
            self.last_evaluation_progress = float('-inf')
            workload.evaluation_progress = self.evaluation_progress if workload.rank == 0 else None
        except Exception:
            if self.service is not None:
                try:self.service.close()
                except Exception:pass
            raise

    def _collective(self, call):
        result=[None,None]
        if self.workload.rank==0:
            try:result[0]=call()
            except Exception as exc:result[1]=f'{type(exc).__name__}: {exc}'
        if self.workload.world_size>1:dist.broadcast_object_list(result,src=0)
        if result[1] is not None:raise RuntimeError(f'production operations failure: {result[1]}')
        return result[0]

    def poll(self, *, force=False):
        w=self.workload
        if not force and w.state['updates']%self.config.health_interval_updates:return
        def inspect():
            # Re-read identity so replacing an allocation file cannot reset the budget.
            if load_allocation(self.config.allocation_record)!=self.allocation:
                raise ValueError('allocation record changed while running')
            budget=remaining(self.allocation,reserve_seconds=self.config.shutdown_reserve_seconds)
            if w.checkpoint_writer is not None:w.checkpoint_writer.check()
            self.service.scan()
            write_json_atomic(w.output_dir/'heartbeat.json',{'schema':1,'time_unix':time.time(),
                'state':dict(w.state),'allocation_identity':self.allocation['identity_sha256'],
                'budget':budget,'status':'stopping' if budget['training_seconds']<=0 else 'running'})
            return budget['training_seconds']<=0
        if self._collective(inspect):w.should_stop[0]=True

    def evaluation_progress(self, progress):
        """Heartbeat only after real evaluation work; no timer that hides a hang."""
        if self.closed:
            return
        now = time.monotonic()
        if now-self.last_evaluation_progress < 30:
            return
        w = self.workload
        if w.device.type == 'cuda':
            import torch
            torch.cuda.synchronize(w.device)
        budget = remaining(self.allocation, reserve_seconds=self.config.shutdown_reserve_seconds)
        if budget['training_seconds'] <= 0:
            raise TimeoutError('allocation training deadline reached during evaluation')
        write_json_atomic(w.output_dir/'heartbeat.json', dict(schema=1, time_unix=time.time(),
            state=dict(w.state), allocation_identity=self.allocation['identity_sha256'], budget=budget,
            status='evaluating', progress=dict(progress)))
        self.last_evaluation_progress = now

    def close_local(self):
        if self.closed:return
        self.closed=True
        self.workload.evaluation_progress = None
        if self.workload.rank==0:
            error=None
            try:
                if self.workload.checkpoint_writer is not None:self.workload.checkpoint_writer.flush()
                if self.service is not None:self.service.scan()
            except Exception as exc:error=exc
            if self.service is not None:
                try:
                    self.service.close()
                    latest=self.workload.output_dir/'checkpoints/latest.json'
                    if latest.exists() and not verify_durable(self.service.store,json.loads(latest.read_text())):
                        raise RuntimeError('latest checkpoint is not durable at exit')
                except Exception as exc:
                    if error is None:error=exc
            if error is not None:raise error

    def close(self):
        self._collective(self.close_local)
