"""CPU-only fixed-QAT forecast from explicitly supplied sustained node rates."""
import argparse
import json

from production.artifacts import external_path
from terminus.integrity import write_json_atomic
from .allocation import fixed_qat_forecast, load, remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('dense', 'qat'), default='dense')
    parser.add_argument('--token-clock', choices=('continuous', 'scored_targets'), default='continuous',
                        help='Packed runs use scored_targets for every budget and measured rate')
    for name in ('dense-tokens-seen', 'qat-tokens', 'batch-tokens',
                 'dense-cooldown-tokens', 'max-dense-tokens'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--qat-tokens-seen', type=int, default=0)
    parser.add_argument('--dense-warmup-tokens', type=int, default=0)
    for name in ('dense-cluster-tps', 'qat-cluster-tps', 'reserve-seconds'):
        parser.add_argument('--' + name, type=float, required=True)
    args = vars(parser.parse_args())
    record = load(args.pop('allocation'))
    output = external_path(args.pop('output'), fresh=True)
    budget = remaining(record)
    inputs = dict(args, remaining_node_seconds=budget['remaining_node_seconds'])
    result = fixed_qat_forecast(**inputs)
    write_json_atomic(output, {
        'allocation_identity': record['identity_sha256'], 'budget': budget,
        'inputs': inputs, 'forecast': result,
        'status': 'forecast_only; no config, checkpoint or learning policy changed',
    })
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
