"""Prospective WSD horizon/curriculum changes from a stopped committed checkpoint.

This CPU transaction preserves weights, optimizer state and every consumed
example. Native resume still checks the inherited runtime and preflight. It
never launches training, resets warmup, or moves an already committed ending.
"""
import argparse
import copy
from dataclasses import asdict
import fcntl
import json
from pathlib import Path

from production.artifacts import external_path
from terminus.checkpoint import commit_checkpoint_payload, load_checkpoint
from terminus.data import TokenManifest
from terminus.integrity import fingerprint, source_hashes, validate_manifest, write_json_atomic
from terminus.stream_contract import verify_training_continuation, verify_packed_continuation
from terminus.train import OPERATIONAL_TRAINING_FIELDS, load_config
from .staging import verify as verify_staging


def schedule_proof(parent, child_training, state, old_total, new_total, *, old_batch, new_batch, weight_mode,
                   packed=False):
    old=dict(parent); new=asdict(child_training)
    for name in OPERATIONAL_TRAINING_FIELDS:new.pop(name)
    if old.get('schedule')!='wsd' or new.get('schedule')!='wsd':
        raise ValueError('prospective horizon changes require WSD')
    cursor = state.get('input_slots_seen') if packed else state['tokens_seen']
    if old_batch!=new_batch or (not packed and new_total%new_batch) or cursor!=state['updates']*old_batch:
        raise ValueError('horizon changes require the same complete global update')
    phase=old.get('phase_start_tokens') or 0
    commitment=state.get('ending_commitment',{})
    if commitment.get('weight_mode')==weight_mode and commitment.get('phase_start_tokens')==phase:
        raise ValueError('phase ending is already committed')
    old_decay=old.pop('decay_tokens'); new_decay=new.pop('decay_tokens')
    old_accum=old.pop('gradient_accumulation_steps');new_accum=new.pop('gradient_accumulation_steps')
    if old!=new:raise ValueError('horizon fork changed a fixed learning policy, including warmup')
    seen=state['tokens_seen']
    if (new_total<=seen or old_total<=seen or seen>=old_total-old_decay
            or seen>=new_total-new_decay or new_total-phase<new['warmup_tokens']+new_decay):
        raise ValueError('cannot move a started cooldown or place a new cooldown in consumed history')
    if state['next_micro_step']!=state['updates']*old_accum:
        raise ValueError('parent microstep cursor is inconsistent')
    # Both schedules have identical warmup and peak, and all consumed updates
    # precede either cooldown. This proves the complete LR prefix without a
    # million-update Python loop. Microbatch regrouping preserves global steps.
    return {'past_lr_identical_updates':state['updates'], 'warmup_tokens':new['warmup_tokens'],
            'old_total_tokens':old_total,'new_total_tokens':new_total,
            'old_decay_tokens':old_decay,'new_decay_tokens':new_decay,
            'new_next_micro_step':state['updates']*new_accum}


def fork(parent_run, parent_manifest_path, child_config_path, child_manifest_path, data_root,
         output_dir, *, parent_checkpoint=None, child_staging_receipt=None,
         data_policy='continue', commit_ending=False):
    parent_run=external_path(parent_run);output=external_path(output_dir,fresh=True)
    parent_manifest_path=external_path(parent_manifest_path)
    child_config_path=external_path(child_config_path);child_manifest_path=external_path(child_manifest_path)
    data_root=external_path(data_root)
    if data_policy not in ('continue','phase_mix'):raise ValueError('explicit continuation policy required')
    with (parent_run/'.run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=json.loads((parent_run/'run.json').read_text());parent=run['contract']
        if fingerprint(parent)!=run['contract_sha256']:raise ValueError('parent run identity mismatch')
        if parent['source']!=source_hashes():raise ValueError('source changed; horizon forks cannot authorize a code migration')
        payload=load_checkpoint(parent_checkpoint or parent_run/'checkpoints',parent)
        state=copy.deepcopy(payload['state'])
        model,training,optimizer=load_config(child_config_path)
        if asdict(model)!=parent['model'] or json.loads(json.dumps(asdict(optimizer)))!=parent['optimizer']:
            raise ValueError('horizon fork must preserve architecture, numerical recipe and peak optimizer rates')
        pm=TokenManifest.load(parent_manifest_path);cm=TokenManifest.load(child_manifest_path)
        if fingerprint(asdict(pm))!=parent['data']['manifest_sha256']:
            raise ValueError('parent manifest does not match checkpoint')
        if pm.tokenizer!=cm.tokenizer or [asdict(s) for s in pm.shards_for(split=pm.config.val_split)]!=[asdict(s) for s in cm.shards_for(split=cm.config.val_split)]:
            raise ValueError('horizon fork changed tokenizer or held-out data')
        if pm.is_packed != cm.is_packed:
            raise ValueError('horizon fork cannot change the packing contract')
        if pm.is_packed and pm.provenance['packed_stream']['val'] != cm.provenance['packed_stream']['val']:
            raise ValueError('horizon fork changed held-out document metadata or weights')
        world=parent['world_size']
        if len(payload['rank_rng'])!=world:raise ValueError('rank RNG state count mismatch')
        proof=schedule_proof(parent['training'],training,state,pm.scored_train_tokens,cm.scored_train_tokens,
            old_batch=pm.config.tokens_per_step*world*parent['training']['gradient_accumulation_steps'],
            new_batch=cm.config.tokens_per_step*world*training.gradient_accumulation_steps,
            weight_mode=model.weight_mode,packed=pm.is_packed)
        if child_staging_receipt:
            report=verify_staging(json.loads(external_path(child_staging_receipt).read_text()),cm,data_root,model,
                                  development_mode=training.development_mode)
        else:
            report=validate_manifest(cm,data_root,model,development_mode=training.development_mode)
        if pm.is_packed:
            continuity=verify_packed_continuation(parent['data']['training_stream'],report['training_stream'],
                                                state,policy=data_policy)
        else:
            continuity=verify_training_continuation(parent['data']['training_stream'],report['training_stream'],
                                                   state['tokens_seen'],policy=data_policy)
        state['next_micro_step']=proof['new_next_micro_step']
        if commit_ending:
            state['ending_commitment']={'weight_mode':model.weight_mode,'phase_start_tokens':training.phase_start_tokens or 0,
                                        'total_tokens':cm.scored_train_tokens,'decay_tokens':training.decay_tokens}
        settings=asdict(training)
        for name in OPERATIONAL_TRAINING_FIELDS:settings.pop(name)
        # Match the trainer's portable identity; local inode/cache stats belong
        # to the staging receipt and must not enter a checkpoint contract.
        child=parent|{'training':settings,'data':{k:v for k,v in report.items() if k!='packed_files'}}
        # Only metadata changes. Native tensors/optimizer/RNG are carried as CPU
        # checkpoint tensors and checked against the same runtime on resumption.
        child_payload=payload|{'contract':child,'state':state}
        output.mkdir(parents=True)
        write_json_atomic(output/'config.json',json.loads(child_config_path.read_text()))
        write_json_atomic(output/'manifest.json',asdict(cm))
        checkpoint=commit_checkpoint_payload(output/'checkpoints',child_payload)
        write_json_atomic(output/'run.json',{'contract':child,'contract_sha256':fingerprint(child)})
        receipt={'schema':1,'event':'prospective_wsd_fork','parent_run':str(parent_run),
                 'parent_contract_sha256':run['contract_sha256'],'child_contract_sha256':fingerprint(child),
                 'checkpoint':checkpoint.name,'state':state,'schedule_proof':proof,
                 'stream_proof':continuity,'commit_ending':commit_ending,
                 'runtime':'inherited; must match native runtime at resume'}
        write_json_atomic(output/'horizon_receipt.json',receipt)
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run','parent-manifest','child-config','child-manifest','data-root','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--parent-checkpoint');p.add_argument('--child-staging-receipt')
    p.add_argument('--data-policy',choices=('continue','phase_mix'),default='continue')
    p.add_argument('--commit-ending',action='store_true')
    a=p.parse_args()
    result=fork(a.parent_run,a.parent_manifest,a.child_config,a.child_manifest,a.data_root,a.output_dir,
                parent_checkpoint=a.parent_checkpoint,child_staging_receipt=a.child_staging_receipt,
                data_policy=a.data_policy,commit_ending=a.commit_ending)
    print(json.dumps(result))


if __name__=='__main__':main()
