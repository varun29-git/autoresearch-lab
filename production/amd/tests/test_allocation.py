import json
from dataclasses import asdict
import sys
import time

import pytest

from production.amd.allocation import create, load, remaining, fixed_qat_forecast
from production.amd.supervisor import Policy, retry_allowed, watchdog_reason, supervise


def test_allocation_counts_idle_and_restarts_and_cannot_be_reinitialized(tmp_path):
    path=tmp_path/'allocation.json'
    record=create(path,allocation_id='node-1',started_unix=1000,gpu_hours=5760,gpus=8)
    assert remaining(load(path),now=1000+3600)['elapsed_gpu_hours']==8
    assert remaining(load(path),now=1000+2*3600)['remaining_gpu_hours']==5744
    with pytest.raises(FileExistsError):create(path,allocation_id='reset',started_unix=2000)
    changed=json.loads(path.read_text());changed['started_unix']=2000;path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):load(path)


def test_faster_dense_rate_adds_only_dense_tokens_and_reserves_qat():
    args=dict(remaining_node_seconds=100,dense_tokens_seen=1000,qat_tokens=2000,
              qat_cluster_tps=100,batch_tokens=100,dense_cooldown_tokens=500,
              reserve_seconds=10,max_dense_tokens=100000)
    a=fixed_qat_forecast(dense_cluster_tps=100,**args)
    b=fixed_qat_forecast(dense_cluster_tps=200,**args)
    assert a['dense_end_tokens']==8000 and b['dense_end_tokens']==15000
    assert a['qat_tokens']==b['qat_tokens']==2000
    assert a['estimated_remaining_training_seconds']==90
    with pytest.raises(ValueError,match='cannot fund'):
        fixed_qat_forecast(dense_cluster_tps=100,**(args|{'remaining_node_seconds':32}))
    q=fixed_qat_forecast(dense_cluster_tps=100,**(args|{'phase':'qat','qat_tokens_seen':1000}))
    assert q['remaining_dense_tokens']==0 and q['qat_tokens_remaining']==1000


def policy(allocation):
    return Policy(str(allocation),[sys.executable,'-c','pass'],[sys.executable,'-c','pass'])


def test_retry_policy_rejects_numeric_data_oom_and_unknown_failures(tmp_path):
    p=policy(tmp_path/'allocation.json')
    for name in ('FloatingPointError','ValueError','OutOfMemoryError','RuntimeError'):
        assert not retry_allowed(1,{'error_type':name},restarts=0,retry_seconds=0,policy=p)
    assert retry_allowed(1,{'error_type':'DistNetworkError'},restarts=0,retry_seconds=0,policy=p)
    assert not retry_allowed(75,None,restarts=2,retry_seconds=0,policy=p)
    assert not retry_allowed(75,None,restarts=0,retry_seconds=1800,policy=p)


def test_watchdog_has_independent_hard_deadline_and_heartbeat(tmp_path):
    record=create(tmp_path/'allocation.json',allocation_id='node',started_unix=1000,gpu_hours=8,gpus=8)
    p=policy(tmp_path/'allocation.json')
    live={'status':'running','allocation_identity':record['identity_sha256'],'time_unix':2000}
    assert watchdog_reason(record,live,p,2000) is None
    assert watchdog_reason(record,live,p,3000)=='controller_heartbeat_stale'
    assert watchdog_reason(record,None,p,record['deadline_unix']-100)=='hard_allocation_deadline'


def test_cpu_supervisor_calls_explicit_provider_hooks_on_child_failure(tmp_path):
    allocation=tmp_path/'allocation.json'
    create(allocation,allocation_id='test',started_unix=time.time()-1,gpu_hours=1,gpus=1)
    p=policy(allocation)
    cfg=asdict(p)|{'poll_seconds':.01}
    path=tmp_path/'policy.json';path.write_text(json.dumps(cfg))
    root=tmp_path/'run'
    result=supervise([sys.executable,'-c','raise SystemExit(2)'],root,path,spawn_watchdog=False)
    assert result['status']=='failed'
    assert (root/'supervision/provider_stopped.json').exists()
    with pytest.raises(ValueError,match='already stopped'):
        supervise([sys.executable,'-c','raise SystemExit(2)'],root,path,spawn_watchdog=False)


def test_log_storm_stops_child_and_calls_provider_hooks(tmp_path):
    allocation=tmp_path/'allocation.json'
    create(allocation,allocation_id='test',started_unix=time.time()-1,gpu_hours=1,gpus=1)
    cfg=asdict(policy(allocation))|{'poll_seconds':.01,'max_attempt_log_bytes':100}
    path=tmp_path/'policy.json';path.write_text(json.dumps(cfg))
    result=supervise([sys.executable,'-c','import time; print("x"*1000,flush=True); time.sleep(10)'],
                     tmp_path/'run',path,spawn_watchdog=False)
    assert result['status']=='failed' and result['reason']=='attempt_log_limit'
    assert (tmp_path/'run/supervision/provider_stopped.json').exists()
