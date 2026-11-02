"""Persistent aggregate-GPU-hour accounting and fixed-QAT forecasts."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

from production.artifacts import external_path
from terminus.integrity import fingerprint, write_json_atomic


def finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def create(path, *, allocation_id, started_unix, gpu_hours=5760., gpus=8):
    path = external_path(path, fresh=True)
    if not isinstance(allocation_id,str) or not allocation_id.strip() or type(gpus) is not int or gpus <= 0:
        raise ValueError('allocation needs an ID and positive integer GPU count')
    finite_positive(started_unix,'started_unix');finite_positive(gpu_hours,'gpu_hours')
    record={'schema':1,'allocation_id':allocation_id,'started_unix':started_unix,
            'gpu_hours':gpu_hours,'gpus':gpus,'deadline_unix':started_unix+gpu_hours/gpus*3600}
    record['identity_sha256']=fingerprint(record)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents a repeated init from resetting the clock.
    with path.open('x') as handle:
        json.dump(record,handle,sort_keys=True,allow_nan=False);handle.write('\n')
        handle.flush()
        import os
        os.fsync(handle.fileno())
    from terminus.integrity import fsync_directory
    fsync_directory(path.parent)
    return record


def load(path):
    record=json.loads(external_path(path).read_text())
    expected={k:v for k,v in record.items() if k!='identity_sha256'}
    if record.get('schema')!=1 or record.get('identity_sha256')!=fingerprint(expected):
        raise ValueError('allocation identity changed')
    finite_positive(record['started_unix'],'started_unix');finite_positive(record['gpu_hours'],'gpu_hours')
    if type(record['gpus']) is not int or record['gpus']<=0:
        raise ValueError('invalid allocation GPU count')
    if record['deadline_unix']!=record['started_unix']+record['gpu_hours']/record['gpus']*3600:
        raise ValueError('allocation deadline is inconsistent')
    return record


def remaining(record, *, now=None, reserve_seconds=0.):
    now=time.time() if now is None else now
    if not math.isfinite(now) or now < record['started_unix']:
        raise ValueError('allocation start is in the future or clock moved backwards')
    if not math.isfinite(reserve_seconds) or reserve_seconds<0:
        raise ValueError('invalid finishing reserve')
    seconds=max(0.,record['deadline_unix']-now)
    return {'elapsed_gpu_hours':(now-record['started_unix'])*record['gpus']/3600,
            'remaining_gpu_hours':seconds*record['gpus']/3600,
            'remaining_node_seconds':seconds,
            'training_seconds':max(0.,seconds-reserve_seconds)}


def fixed_qat_forecast(*, remaining_node_seconds, dense_tokens_seen, dense_cluster_tps,
                       qat_tokens, qat_tokens_seen=0, qat_cluster_tps,
                       batch_tokens, dense_cooldown_tokens, reserve_seconds,
                       max_dense_tokens, phase='dense', dense_warmup_tokens=0):
    """Rates include normal cadence. Reserve only additional finishing/outage work."""
    for name,value in [('dense_cluster_tps',dense_cluster_tps),('qat_cluster_tps',qat_cluster_tps)]:
        finite_positive(value,name)
    for name,value in [('dense_tokens_seen',dense_tokens_seen),('qat_tokens',qat_tokens),
                       ('qat_tokens_seen',qat_tokens_seen),('dense_cooldown_tokens',dense_cooldown_tokens),
                       ('max_dense_tokens',max_dense_tokens),('dense_warmup_tokens',dense_warmup_tokens)]:
        if type(value) is not int or value<0:raise ValueError(f'{name} must be a nonnegative integer')
    if type(batch_tokens) is not int or batch_tokens<=0:raise ValueError('invalid global batch')
    if (phase not in ('dense','qat') or qat_tokens_seen>qat_tokens or max_dense_tokens<dense_tokens_seen
            or any(v%batch_tokens for v in (dense_tokens_seen,qat_tokens,qat_tokens_seen,dense_cooldown_tokens,max_dense_tokens))):
        raise ValueError('phase budgets must align with complete global updates')
    if (not math.isfinite(remaining_node_seconds) or not math.isfinite(reserve_seconds)
            or remaining_node_seconds<0 or reserve_seconds<0):raise ValueError('invalid remaining time')
    if phase=='dense' and qat_tokens_seen:raise ValueError('QAT progress cannot precede dense completion')
    useful=remaining_node_seconds-reserve_seconds
    qat_seconds=(qat_tokens-qat_tokens_seen)/qat_cluster_tps
    if phase=='qat':
        dense_end=dense_tokens_seen;needed=qat_seconds
    else:
        available=useful-qat_seconds
        possible=max(0,math.floor(available*dense_cluster_tps/batch_tokens))*batch_tokens
        dense_end=min(max_dense_tokens,dense_tokens_seen+possible)
        needed=(dense_end-dense_tokens_seen)/dense_cluster_tps+qat_seconds
        if dense_end-dense_tokens_seen<max(0,dense_warmup_tokens-dense_tokens_seen)+dense_cooldown_tokens:
            raise ValueError('budget cannot fund dense cooldown plus fixed QAT and reserves')
    if needed>useful or useful<0:raise ValueError('budget cannot fund mandatory remaining phases')
    return {'dense_end_tokens':dense_end,'qat_tokens':qat_tokens,'qat_tokens_remaining':qat_tokens-qat_tokens_seen,
            'remaining_dense_tokens':dense_end-dense_tokens_seen,'total_final_tokens':dense_end+qat_tokens,
            'estimated_remaining_training_seconds':needed,'reserve_seconds':reserve_seconds,
            'slack_seconds':useful-needed,'rate_units':'global tokens per node-second; normal cadence included'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--allocation-id',required=True)
    p.add_argument('--started-unix',type=float,required=True,help='Actual provider allocation time, not preparation time')
    p.add_argument('--gpu-hours',type=float,default=5760.);p.add_argument('--gpus',type=int,default=8)
    args=p.parse_args()
    result=create(args.output,allocation_id=args.allocation_id,started_unix=args.started_unix,
                  gpu_hours=args.gpu_hours,gpus=args.gpus)
    print(json.dumps(result))


if __name__=='__main__':main()
