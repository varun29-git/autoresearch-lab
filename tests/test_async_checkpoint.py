import json
import threading
from unittest.mock import patch

import pytest
import torch

from terminus.async_checkpoint import AsyncCheckpointWriter
from terminus.checkpoint import load_checkpoint, rng_state
from terminus.train import run_training
from tests.helpers import fixture, checkpoint_payload


def test_background_snapshot_never_reads_mutating_live_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 3)).sum().backward(); optimizer.step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    expected_moment = optimizer.state[model.weight]['exp_avg'].clone()
    state = {'updates': 1, 'tokens_seen': 8, 'next_micro_step': 1}
    entered, release = threading.Event(), threading.Event()
    from terminus.async_checkpoint import commit_checkpoint_payload
    def paused(directory, payload):
        entered.set()
        assert release.wait(10)
        return commit_checkpoint_payload(directory, payload)
    with patch('terminus.async_checkpoint.commit_checkpoint_payload', paused):
        writer = AsyncCheckpointWriter(tmp_path)
        writer.submit(model, optimizer, state, {}, [rng_state()])
        assert entered.wait(10)
        with torch.no_grad():
            model.weight.add_(100)
            optimizer.state[model.weight]['exp_avg'].add_(200)
        state['updates'] = 999
        release.set(); writer.close()
    saved = load_checkpoint(tmp_path, {})
    assert saved['state']['updates'] == 1
    for k,v in expected.items(): torch.testing.assert_close(saved['model'][k], v)
    torch.testing.assert_close(saved['optimizer']['state'][0]['exp_avg'], expected_moment)


def test_async_failure_preserves_previous_commit_and_propagates(tmp_path):
    model = torch.nn.Linear(3, 2); optimizer = torch.optim.AdamW(model.parameters())
    writer = AsyncCheckpointWriter(tmp_path)
    state = {'updates': 1, 'tokens_seen': 8, 'next_micro_step': 1}
    writer.submit(model, optimizer, state, {}, [rng_state()]); writer.flush()
    with torch.no_grad(): model.weight.fill_(float('nan'))
    writer.submit(model, optimizer, {**state, 'updates': 2}, {}, [rng_state()])
    with pytest.raises(FloatingPointError): writer.flush()
    assert load_checkpoint(tmp_path, {})['state']['updates'] == 1
    with pytest.raises(FloatingPointError): writer.check()


def test_async_training_exact_resume_and_synchronous_equivalence(tmp_path):
    data = tmp_path / 'data'; fixture(data, examples=12, dropout=.1)
    cfg = data / 'config.json'
    args = (cfg, data / 'manifest.json', data)
    run_training(*args, tmp_path / 'sync')
    config = json.loads(cfg.read_text()); config['training']['async_checkpoint'] = True
    cfg.write_text(json.dumps(config))
    run_training(*args, tmp_path / 'async', stop_after_updates=1)
    run_training(*args, tmp_path / 'async', resume=tmp_path / 'async/checkpoints')
    a,b=checkpoint_payload(tmp_path/'sync'),checkpoint_payload(tmp_path/'async')
    assert a['state'] == b['state']
    for key in a['model']: torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    for param,states in a['optimizer']['state'].items():
        for key,value in states.items():
            torch.testing.assert_close(value,b['optimizer']['state'][param][key],rtol=0,atol=0)
