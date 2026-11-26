"""Stage and validate the isolated 0.002-versus-0.001 dense embedding check.

Does not promote rates, alter the frozen H100 matrix, provision hardware or
implement mid-run LR changes. Native execution requires a human-launched H100,
an open allocation ledger and fresh runtime/preflight/data evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
import math
import os
from pathlib import Path

from production.artifacts import external_path
from production.h100_24h import campaign
from production.h100_24h.builder import build_confirmation_config, build_trial_config
from terminus.data import TokenManifest
from terminus.integrity import file_sha256, fingerprint, write_json_atomic
from terminus.train import load_config


ARMS = {'reference': 0.002, 'challenger': 0.001}
GEOMETRIES = ('proxy', 'full_size')


def config_for(geometry, arm, *, microbatch):
    if geometry not in GEOMETRIES or arm not in ARMS:
        raise ValueError('only proxy/full_size and reference/challenger are permitted')
    config = (build_trial_config('H', microbatch=microbatch) if geometry == 'proxy'
              else build_confirmation_config('H', microbatch=microbatch))
    width = config['model']['dim']
    # Production rates are already transferred at width 2304. Invert that
    # transfer ONCE to stage the width-768 proxy: 0.001 -> 0.003.
    config['optimizer']['adamw_lr'] = float(Decimal(str(ARMS[arm])) * 2304 / width)
    return config


def _source():
    return campaign.qualification_source_hashes() | {
        'production/embedding_lr_check.py': file_sha256(__file__)}


def _schedule(geometry):
    return campaign.PROXY_SCHEDULE if geometry == 'proxy' else campaign.CONFIRMATION_SCHEDULE


def validate(packet_dir):
    from terminus.preflight import count_model
    root = external_path(packet_dir)
    packet = json.loads((root / 'packet.json').read_text())
    if (packet.get('schema') != 1 or packet.get('kind') != 'embedding_lr_001_check'
            or packet.get('packet_sha256') != fingerprint({k: v for k, v in packet.items()
                                                          if k != 'packet_sha256'})):
        raise ValueError('invalid embedding LR packet identity')
    if packet.get('source') != _source():
        raise ValueError('source changed after the LR check was staged')
    expected_files = {f'{geometry}/{name}' for geometry in GEOMETRIES
                      for name in ['manifest.json', 'reference.json', 'challenger.json']}
    if set(packet.get('files', {})) != expected_files:
        raise ValueError('packet file set differs from the fixed two-arm check')
    results = {}
    for relative, digest in packet['files'].items():
        if file_sha256(root / relative) != digest:
            raise ValueError(f'packet file changed: {relative}')
    for geometry in GEOMETRIES:
        manifest = TokenManifest.load(root / geometry / 'manifest.json')
        if manifest.config.seq_len != 4096 or manifest.config.usable_train_tokens != _schedule(geometry)['total_tokens']:
            raise ValueError('manifest context/horizon differs from the matched qualification schedule')
        configs = {}
        for arm in ARMS:
            path = root / geometry / f'{arm}.json'
            config = json.loads(path.read_text())
            expected = config_for(geometry, arm, microbatch=manifest.config.batch_size)
            if config != expected:
                raise ValueError(f'{geometry}/{arm} differs from the isolated optimizer comparison')
            model, training, optimizer = load_config(path)
            if manifest.config.tokens_per_step * training.gradient_accumulation_steps != campaign.GLOBAL_BATCH_TOKENS:
                raise ValueError('global batch differs from 2,097,152 tokens')
            counts = count_model(model)
            expected_count = campaign.PROXY_PARAMETER_COUNT if geometry == 'proxy' else campaign.FULL_PARAMETER_COUNT
            if counts['total'] != expected_count:
                raise ValueError('model parameter count differs from the selected geometry')
            configs[arm] = config
        a, b = configs['reference'], configs['challenger']
        comparable = json.loads(json.dumps(b))
        comparable['optimizer']['adamw_lr'] = a['optimizer']['adamw_lr']
        if comparable != a:
            raise ValueError('an additional setting changed between LR arms')
        results[geometry] = {'parameters': expected_count, 'world_size': 1,
                             'global_batch_tokens': campaign.GLOBAL_BATCH_TOKENS,
                             'tokens_per_arm': _schedule(geometry)['total_tokens'],
                             'peak_embedding_lrs': {arm: c['optimizer']['adamw_lr'] for arm, c in configs.items()}}
    return {'status': 'offline_contract_checks_passed', 'geometries': results,
            'native_run_performed': False, 'production_lr_changed': False}


def stage(*, proxy_manifest, full_manifest, output_dir):
    root = external_path(output_dir, fresh=True)
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    for geometry, source in [('proxy', proxy_manifest), ('full_size', full_manifest)]:
        source = external_path(source)
        manifest = TokenManifest.load(source)
        directory = root / geometry
        write_json_atomic(directory / 'manifest.json', asdict(manifest))
        for arm in ARMS:
            write_json_atomic(directory / f'{arm}.json', config_for(geometry, arm,
                              microbatch=manifest.config.batch_size))
        for path in directory.iterdir():
            files[str(path.relative_to(root))] = file_sha256(path)
    packet = {'schema': 1, 'kind': 'embedding_lr_001_check', 'source': _source(), 'files': files,
              'production_equivalent_lrs': ARMS, 'selection_margins': campaign.SELECTION_MARGINS,
              'global_batch_tokens': campaign.GLOBAL_BATCH_TOKENS, 'world_size': 1,
              'production_lr_changed': False, 'native_run_performed': False,
              'notes': ['Cold-start matched comparison; not an existing-checkpoint LR intervention.',
                        'Only embedding/head AdamW changes; Muon/norms/decay/seed/data/schedule are fixed.',
                        'Qualification schedules must not replace the production schedules.',
                        'Historical H results are background; fresh matched results require matching source/runtime/data.',
                        'Full-size pair is optional after proxy screening; AMD acceptance remains separate.',
                        'These limits do not authorize another paid allocation or reset existing budget usage.']}
    packet['packet_sha256'] = fingerprint(packet)
    write_json_atomic(root / 'packet.json', packet)
    receipt = validate(root)
    write_json_atomic(root / 'offline-validation.json', receipt)
    return receipt


def run(args):
    import torch
    from production.h100_24h import budget
    from production.h100_24h.recovery import BackupQueue, destination, verify_remote
    from production.h100_24h.runner import assert_effective_groups, drive
    from production.runtime import capture_runtime, verify_runtime
    from terminus.train import prepare_training

    root = external_path(args.packet)
    receipt = validate(root)
    if (int(os.environ.get('WORLD_SIZE', '1')) != 1 or not torch.cuda.is_available()
            or torch.version.hip or 'H100' not in torch.cuda.get_device_name(0)):
        raise ValueError('this isolated check requires exactly one human-launched H100')
    if (not math.isfinite(args.forecast_hours) or args.forecast_hours <= 0
            or not math.isfinite(args.future_work_hours) or args.future_work_hours < 0):
        raise ValueError('a positive measured stage forecast is required')
    ledger = budget.Ledger(external_path(args.ledger))
    if not ledger.allocation_open:
        raise ValueError('human must record actual VM billing start in the allocation ledger')
    ledger.require(args.forecast_hours + args.future_work_hours,
                   label='current and remaining embedding LR work', reserve_hours=0.5)
    verify_runtime(json.loads(external_path(args.runtime_report).read_text()), capture_runtime())
    geometry, arm = args.geometry, args.arm
    config = root / geometry / f'{arm}.json'
    manifest = root / geometry / 'manifest.json'
    output = root / geometry / 'runs' / arm
    output.mkdir(parents=True, exist_ok=True)
    if (not args.backup_destination.startswith('s3://') or args.backup_max_pending_bytes <= 0
            or not math.isfinite(args.backup_max_pending_seconds) or args.backup_max_pending_seconds <= 0
            or args.min_free_bytes <= 0):
        raise ValueError('explicit S3 backup and positive measured queue/disk limits are required')
    packet_identity = json.loads((root / 'packet.json').read_text())['packet_sha256']
    # All arms produce identical checkpoint basenames. Keep their remote
    # objects separate even when the operator supplies one campaign prefix.
    remote_prefix = f"{args.backup_destination.rstrip('/')}/{packet_identity}/{geometry}/{arm}"
    backup = BackupQueue(output / 'checkpoints', destination(remote_prefix))
    backup_bounds = {'max_pending_bytes': args.backup_max_pending_bytes,
                     'max_pending_seconds': args.backup_max_pending_seconds,
                     'min_free_bytes': args.min_free_bytes}
    ledger.log_activity('embedding_lr_check_start', {'geometry': geometry, 'arm': arm})
    try:
        with prepare_training(config, manifest, external_path(args.data_root), output,
                              resume=args.resume, preflight_report=external_path(args.preflight_report)) as workload:
            receipt['effective_groups'] = assert_effective_groups(workload.model_config, workload.optimizer,
                expected_optimizer=config_for(geometry, arm,
                    microbatch=workload.manifest.config.batch_size)['optimizer'])
            write_json_atomic(output / 'qualification-gate.json', receipt)
            schedule = _schedule(geometry)
            summary = drive(workload, boundaries=tuple(schedule['boundary_updates']), ledger_path=args.ledger,
                            final_update=schedule['total_updates'], output_dir=output,
                            future_work_hours=args.future_work_hours,
                            backup=backup, backup_bounds=backup_bounds)
            completed = workload.state['tokens_seen'] == workload.total_tokens
        write_json_atomic(output / 'summary.json', {'geometry': geometry, 'arm': arm, 'complete': completed,
                                                   'summary': summary, 'production_lr_changed': False})
        return 0 if completed and not summary['alarms'] else 2
    finally:
        try:
            status = backup.close()
            if status['failed']:
                raise RuntimeError('checkpoint upload failed')
            latest_path = output / 'checkpoints/latest.json'
            if latest_path.exists():
                latest = json.loads(latest_path.read_text())
                remote = verify_remote(backup.dest, latest['file'], expected_sha256=latest['sha256'])
                if not remote['durable']:
                    raise RuntimeError('latest LR-check checkpoint is not durable')
        finally:
            budget.Ledger(external_path(args.ledger)).log_activity('embedding_lr_check_end',
                                                                 {'geometry': geometry, 'arm': arm})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('stage')
    s.add_argument('--proxy-manifest', required=True)
    s.add_argument('--full-manifest', required=True)
    s.add_argument('--output-dir', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--packet', required=True)
    r = sub.add_parser('run', help='Native execution only after a human GPU launch and acceptance gates')
    r.add_argument('--packet', required=True)
    r.add_argument('--geometry', choices=GEOMETRIES, required=True)
    r.add_argument('--arm', choices=tuple(ARMS), required=True)
    for name in ['data-root', 'runtime-report', 'preflight-report', 'ledger', 'forecast-hours']:
        r.add_argument('--' + name, required=True, type=float if name == 'forecast-hours' else str)
    r.add_argument('--resume')
    r.add_argument('--future-work-hours', required=True, type=float,
                   help='Measured forecast of all remaining arms; reserve it within the same ledger')
    r.add_argument('--backup-destination', required=True)
    r.add_argument('--backup-max-pending-bytes', required=True, type=int)
    r.add_argument('--backup-max-pending-seconds', required=True, type=float)
    r.add_argument('--min-free-bytes', required=True, type=int)
    args = p.parse_args()
    if args.mode == 'stage':
        print(json.dumps(stage(proxy_manifest=args.proxy_manifest, full_manifest=args.full_manifest,
                               output_dir=args.output_dir), indent=2))
    elif args.mode == 'validate':
        print(json.dumps(validate(args.packet), indent=2))
    else:
        return run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
