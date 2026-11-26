"""Human-launched production supervisor and independent billing watchdog.

No provisioning. Provider stop/verification commands are explicit external argv
arrays; no shell interpolation, inferred resource IDs or embedded credentials.
"""
import argparse
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from production.artifacts import external_path
from terminus.integrity import fingerprint, write_json_atomic
from .allocation import load as load_allocation, remaining


@dataclass(frozen=True)
class Policy:
    allocation_record: str
    provider_stop_command: list
    provider_verify_command: list
    shutdown_reserve_seconds: float = 1800.
    heartbeat_timeout_seconds: float = 900.
    startup_timeout_seconds: float = 1800.
    grace_seconds: float = 1200.
    poll_seconds: float = 5.
    provider_timeout_seconds: float = 60.
    max_restarts: int = 2
    max_retry_seconds: float = 1800.
    restart_delay_seconds: float = 30.
    max_attempt_log_bytes: int = 1024**3
    schema: int = 1

    def __post_init__(self):
        if self.schema!=1 or type(self.max_restarts) is not int or self.max_restarts<0:
            raise ValueError('invalid supervisor schema/restart count')
        if type(self.max_attempt_log_bytes) is not int or self.max_attempt_log_bytes<=0:
            raise ValueError('invalid attempt log byte limit')
        for command in (self.provider_stop_command,self.provider_verify_command):
            if not isinstance(command,list) or not command or any(not isinstance(v,str) or not v for v in command):
                raise ValueError('provider stop and verification need explicit nonempty argv arrays')
        for name in ('shutdown_reserve_seconds','heartbeat_timeout_seconds','startup_timeout_seconds',
                     'grace_seconds','poll_seconds','provider_timeout_seconds','max_retry_seconds','restart_delay_seconds'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'invalid {name}')
        if self.shutdown_reserve_seconds<self.grace_seconds+2*self.provider_timeout_seconds:
            raise ValueError('reserve must fund graceful stop and provider shutdown/verification')


def policy_from(path):
    return Policy(**json.loads(external_path(path).read_text()))


def event(path, **record):
    record={'time_unix':time.time(),**record}
    with Path(path).open('a') as handle:
        handle.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())


def retry_allowed(returncode, incident, *, restarts, retry_seconds, policy):
    if restarts>=policy.max_restarts or retry_seconds>=policy.max_retry_seconds:return False
    # OOM, NaN, invalid data/config and unknown errors are terminal by default.
    if incident:
        return incident.get('error_type') in ('DistNetworkError','DistributedNetworkError','ConnectionResetError')
    return returncode==75  # Explicit EX_TEMPFAIL from a reviewed wrapper only.


def latest_incident(root, since):
    path=Path(root)/'incidents.jsonl'
    if not path.exists():return None
    with path.open('rb') as handle:
        size=path.stat().st_size;handle.seek(max(0,size-65536))
        if handle.tell():handle.readline()
        records=[]
        for line in handle:
            try:records.append(json.loads(line))
            except (ValueError,UnicodeError):pass
    # Core incidents may lack a timestamp; use file mtime to avoid stale failures.
    return records[-1] if records and path.stat().st_mtime>=since else None


def provider_stop(policy, directory):
    """Serialize controller/watchdog stop requests; verify actual provider state."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'provider.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        marker=directory/'provider_stopped.json'
        if marker.exists():return json.loads(marker.read_text())
        results=[]
        for name,command in [('stop',policy.provider_stop_command),('verify',policy.provider_verify_command)]:
            try:
                result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,timeout=policy.provider_timeout_seconds)
                code=result.returncode
            except subprocess.TimeoutExpired:code='timeout'
            except OSError:code='unavailable'
            results.append({'action':name,'returncode':code,'command_sha256':fingerprint(command)})
        report={'status':'verified_stopped' if results[-1]['returncode']==0 else 'stop_unverified',
                'time_unix':time.time(),'actions':results}
        write_json_atomic(directory/'provider_stop_attempt.json',report)
        if report['status']=='verified_stopped':write_json_atomic(marker,report)
        return report


def watchdog_reason(record, status, policy, now):
    if now>=record['deadline_unix']-2*policy.provider_timeout_seconds:return 'hard_allocation_deadline'
    if status is None:return None
    if status.get('allocation_identity')!=record['identity_sha256']:return 'allocation_identity_mismatch'
    if status.get('status') in ('completed','failed','stopped'):return 'controller_terminal'
    if now-status.get('time_unix',0)>policy.heartbeat_timeout_seconds:return 'controller_heartbeat_stale'
    return None


def watchdog(policy_path, root):
    policy=policy_from(policy_path);record=load_allocation(policy.allocation_record)
    directory=external_path(root)/'supervision';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'watchdog.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (directory/'provider_stopped.json').exists():
            path=directory/'status.json'
            try:
                status=json.loads(path.read_text()) if path.exists() else None
                reason=watchdog_reason(record,status,policy,time.time())
            except (ValueError,OSError):
                reason='unreadable_controller_status'
            if reason:
                event(directory/'watchdog.jsonl',event='provider_stop_requested',reason=reason)
                result=provider_stop(policy,directory)
                if result['status']=='verified_stopped':return
            time.sleep(policy.poll_seconds)


def _terminate(process, signum):
    try:os.killpg(process.pid,signum)
    except ProcessLookupError:pass


def supervise(command, root, policy_path, *, spawn_watchdog=True):
    if not isinstance(command,list) or not command or any(not isinstance(v,str) for v in command):
        raise ValueError('training command must be an argv array')
    root=external_path(root);root.mkdir(parents=True,exist_ok=True)
    directory=root/'supervision';directory.mkdir(exist_ok=True)
    policy=policy_from(policy_path);record=load_allocation(policy.allocation_record)
    identity=fingerprint({'policy':json.loads(Path(policy_path).read_text()),'command':command,
                          'allocation':record['identity_sha256']})
    state_path=directory/'controller.json'
    with (directory/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state['identity']!=identity:raise ValueError('controller policy/command/allocation changed')
        else:
            state={'identity':identity,'restarts':0,'retry_seconds':0.,'attempts':0}
            write_json_atomic(state_path,state)
        if (directory/'provider_stopped.json').exists():raise ValueError('allocation is already stopped')
        # Publish before starting the watchdog or the training child.
        status={'status':'starting','time_unix':time.time(),'allocation_identity':record['identity_sha256'],
                'controller_pid':os.getpid()}
        write_json_atomic(directory/'status.json',status)
        if spawn_watchdog:
            subprocess.Popen([sys.executable,'-m','production.amd.supervisor','watchdog',
                              '--policy',str(external_path(policy_path)),'--run-dir',str(root)],
                             stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                             start_new_session=True)
        signal_stop=[None]
        def stop_requested(signum,frame):signal_stop[0]=f'supervisor_signal_{signum}'
        old={s:signal.signal(s,stop_requested) for s in (signal.SIGTERM,signal.SIGINT)}
        process=None;terminal='failed';reason=None
        try:
            while True:
                if signal_stop[0]:
                    reason=signal_stop[0];terminal='stopped';break
                if remaining(record,reserve_seconds=policy.shutdown_reserve_seconds)['training_seconds']<=0:
                    reason='allocation_deadline';terminal='stopped';break
                actual=list(command)
                if state['attempts']:
                    checkpoint=root/'checkpoints'
                    if not (checkpoint/'latest.json').exists():raise RuntimeError('retry has no committed checkpoint')
                    if '--resume' in actual:
                        at=actual.index('--resume')
                        if at+1>=len(actual):raise ValueError('missing initial resume argument')
                        del actual[at:at+2]
                    # Trainer verifies exact payload/contract before touching optimizer state.
                    actual += ['--resume',str(checkpoint)]
                started=time.time();mono=time.monotonic();state['attempts']+=1
                write_json_atomic(state_path,state)
                log_path=directory/f"attempt_{state['attempts']:03d}.log"
                # Log to a regular file: a blocked pipe reader cannot stall training.
                with log_path.open('ab',buffering=0) as log:
                    process=subprocess.Popen(actual,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    event(directory/'events.jsonl',event='attempt_started',pid=process.pid,attempt=state['attempts'])
                    sent=None;reason=None;last_clock=started
                    while process.poll() is None:
                        now=time.time()
                        if now<last_clock-1:reason='clock_moved_backwards'
                        last_clock=max(last_clock,now)
                        if signal_stop[0]:reason=signal_stop[0]
                        if now>=record['deadline_unix']-policy.shutdown_reserve_seconds:reason='allocation_deadline'
                        if (directory/'provider_stopped.json').exists():reason='provider_stopped'
                        if log_path.stat().st_size>policy.max_attempt_log_bytes:reason='attempt_log_limit'
                        heartbeat=root/'heartbeat.json'
                        if heartbeat.exists():
                            beat=json.loads(heartbeat.read_text())
                            if beat.get('allocation_identity')!=record['identity_sha256']:reason='training_allocation_mismatch'
                            elif now-beat.get('time_unix',0)>policy.heartbeat_timeout_seconds and now-started>policy.startup_timeout_seconds:
                                reason='training_heartbeat_stale'
                        elif now-started>policy.startup_timeout_seconds:reason='training_startup_timeout'
                        if reason and sent is None:
                            _terminate(process,signal.SIGTERM);sent=time.monotonic()
                            event(directory/'events.jsonl',event='graceful_stop',reason=reason)
                        if sent is not None and time.monotonic()-sent>=policy.grace_seconds:_terminate(process,signal.SIGKILL)
                        status.update(status='stopping' if reason else 'running',time_unix=now,
                                      child_pid=process.pid,reason=reason,attempt=state['attempts'])
                        write_json_atomic(directory/'status.json',status)
                        time.sleep(policy.poll_seconds)
                    code=process.wait()
                    _terminate(process,signal.SIGKILL)  # No orphan ranks survive into a retry.
                duration=time.monotonic()-mono
                if code==0 and reason is None:terminal='completed';break
                if reason:terminal='stopped' if reason=='allocation_deadline' else 'failed';break
                incident=latest_incident(root,started)
                if state['attempts']>1:
                    state['retry_seconds']+=min(duration,policy.startup_timeout_seconds)
                if not retry_allowed(code,incident,restarts=state['restarts'],retry_seconds=state['retry_seconds'],policy=policy):
                    reason='nonretryable_failure';break
                state['restarts']+=1;write_json_atomic(state_path,state)
                event(directory/'events.jsonl',event='bounded_retry',returncode=code,restarts=state['restarts'])
                time.sleep(policy.restart_delay_seconds)
            write_json_atomic(state_path,state)
            status.update(status=terminal,reason=reason,time_unix=time.time(),returncode=process.returncode if process else None)
            write_json_atomic(directory/'status.json',status)
            return status
        except BaseException as exc:
            status.update(status='failed',reason=type(exc).__name__,time_unix=time.time())
            write_json_atomic(directory/'status.json',status)
            raise
        finally:
            if process is not None:
                _terminate(process,signal.SIGKILL)
                if process.poll() is None:process.wait(timeout=15)
            # Failure of the hook is recorded, not mistaken for stopped billing.
            try:
                report=provider_stop(policy,directory)
                status['provider_stop_status']=report['status']
                write_json_atomic(directory/'status.json',status)
            finally:
                for number,handler in old.items():signal.signal(number,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('run','watchdog'));p.add_argument('--policy',required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--command-json')
    args=p.parse_args()
    if args.mode=='watchdog':watchdog(args.policy,args.run_dir)
    else:
        if not args.command_json:p.error('run requires --command-json')
        result=supervise(json.loads(external_path(args.command_json).read_text()),args.run_dir,args.policy)
        print(json.dumps(result))
        if result['status']!='completed' or result.get('provider_stop_status')!='verified_stopped':raise SystemExit(1)


if __name__=='__main__':main()
