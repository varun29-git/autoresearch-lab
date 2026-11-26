import torch

from terminus.task_eval import score_frozen_choices
from terminus.train import evaluate
from tests.test_packed_acceptance_guards import weighted_fixture


def test_packed_evaluation_reports_completed_batches_without_changing_loss_or_rng(tmp_path):
    model, manifest = weighted_fixture(tmp_path)
    args = (model, manifest, tmp_path, torch.device('cpu'), torch.float32, 2, 0, 1)
    expected = evaluate(*args)
    rng = torch.get_rng_state().clone()
    progress = []
    actual = evaluate(*args, progress_callback=progress.append)
    assert actual == expected
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert {p['key'] for p in progress} == {'en', 'bn'}
    assert all(p['phase'] == 'heldout' and 0 < p['completed_sequences'] <= p['total_sequences'] == 2
               for p in progress)
    assert {p['key'] for p in progress if p['completed_sequences'] == 2} == {'en', 'bn'}


def test_choice_scoring_reports_real_forward_progress_without_changing_scores(tmp_path):
    model, _ = weighted_fixture(tmp_path)
    records = [dict(task='test', item_id=str(i), prompt_ids=[40, 41],
                    choice_ids=[[42, 43], [44]], correct_choice=i) for i in range(2)]
    options = dict(device=torch.device('cpu'), max_seq_len=8)
    expected = score_frozen_choices(model, records, **options)
    progress = []
    actual = score_frozen_choices(model, records, **options, progress_callback=progress.append)
    assert actual == expected and model.training
    assert [p['completed_choices'] for p in progress] == [1, 2, 3, 4]
    assert all(p['phase'] == 'benchmark' and p['total_choices'] == 4 for p in progress)
