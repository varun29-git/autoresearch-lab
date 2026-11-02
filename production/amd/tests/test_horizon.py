from dataclasses import replace
import json

import pytest
import torch

from production.amd.horizon import fork
from terminus.data import TokenManifest
from terminus.train import run_training
from tests.helpers import fixture, checkpoint_payload


def prepare(tmp_path):
    data=tmp_path/'data';fixture(data,examples=64)
    full=TokenManifest.load(data/'manifest.json')
    parent=replace(full,config=replace(full.config,total_train_tokens=256))
    parent.save(data/'parent.json')
    child=replace(full,config=replace(full.config,total_train_tokens=384,batch_size=1))
    child.save(data/'child.json')
    cfg=json.loads((data/'config.json').read_text())
    cfg['model']['weight_mode']='dense'
    cfg['training'].update(schedule='wsd',warmup_tokens=32,decay_tokens=64)
    (data/'parent-config.json').write_text(json.dumps(cfg))
    cfg['training']['gradient_accumulation_steps']=4
    (data/'child-config.json').write_text(json.dumps(cfg))
    run_training(data/'parent-config.json',data/'parent.json',data,tmp_path/'parent',stop_after_updates=2)
    return data


def test_horizon_extension_carries_state_and_rebases_microbatch_cursor(tmp_path):
    data=prepare(tmp_path)
    report=fork(tmp_path/'parent',data/'parent.json',data/'child-config.json',data/'child.json',data,tmp_path/'child')
    a,b=checkpoint_payload(tmp_path/'parent'),checkpoint_payload(tmp_path/'child')
    assert b['state']['tokens_seen']==64 and b['state']['next_micro_step']==8
    assert report['schedule_proof']['past_lr_identical_updates']==2
    assert report['stream_proof']['consumed_tokens']==64
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    for param,states in a['optimizer']['state'].items():
        for key,value in states.items():torch.testing.assert_close(value,b['optimizer']['state'][param][key],rtol=0,atol=0)
    state=run_training(tmp_path/'child/config.json',tmp_path/'child/manifest.json',data,tmp_path/'child',
                       resume=tmp_path/'child/checkpoints',stop_after_updates=3)
    assert state['tokens_seen']==96 and state['next_micro_step']==12


def test_committed_ending_cannot_be_moved_even_before_decay(tmp_path):
    data=prepare(tmp_path)
    fork(tmp_path/'parent',data/'parent.json',data/'child-config.json',data/'child.json',data,tmp_path/'child',commit_ending=True)
    with pytest.raises(ValueError,match='already committed'):
        fork(tmp_path/'child',data/'child.json',data/'child-config.json',data/'child.json',data,tmp_path/'grandchild')


@pytest.mark.parametrize('change',['warmup','lr','consumed_decay','data'])
def test_horizon_cannot_rewrite_past_learning_or_data(tmp_path,change):
    data=prepare(tmp_path)
    cfg=json.loads((data/'child-config.json').read_text())
    if change=='warmup':cfg['training']['warmup_tokens']=64
    if change=='lr':cfg['optimizer']['adamw_lr']*=2
    if change=='consumed_decay':cfg['training']['decay_tokens']=352
    (data/'child-config.json').write_text(json.dumps(cfg))
    if change=='data':
        manifest=TokenManifest.load(data/'child.json')
        manifest=replace(manifest,config=replace(manifest.config,source_start_examples={'en':1}))
        manifest.save(data/'child.json')
    with pytest.raises(ValueError):
        fork(tmp_path/'parent',data/'parent.json',data/'child-config.json',data/'child.json',data,tmp_path/'child')
    assert not (tmp_path/'child/run.json').exists()
