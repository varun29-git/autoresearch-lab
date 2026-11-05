import numpy as np
import pytest

from tests.test_validation_data import make_bundle

def test_scoring_shift_counts_final_guard_and_state(tmp_path):
    import torch
    from production.validation_evaluate import score_cells

    class Model(torch.nn.Module):
        def forward(self, x, targets, return_logits):
            assert return_logits is False
            assert torch.all(targets == x + 1)
            return {"loss_sum": targets.double().sum(), "loss_tokens": targets.numel()}

    root, _, _, _ = make_bundle(tmp_path)
    model = Model()
    result = score_cells(model, root, batch_size=2, weights={"en/web": .5, "hi/web": .5})
    assert result["weighted_nll"] == 108.5
    assert result["per_cell"]["en/web"]["tokens"] == 16
    assert model.training
    with pytest.raises(ValueError, match="final"):
        score_cells(model, root, role="final")
    final = score_cells(model, root, role="final", include_final=True, max_sequences=1)
    assert final["sample_token_weighted_nll"] == 122.5
    assert not final["complete_contract_score"]
    with pytest.raises(ValueError, match="coverage"):
        score_cells(model, root, weights={"en/web": 1.})


def test_real_model_nll_matches_independent_logits_and_batching(tmp_path):
    import torch
    from production.validation_evaluate import score_cells
    from terminus.model import BitNetLlama
    from tests.helpers import tiny_config

    root, manifest, _, _ = make_bundle(tmp_path)
    torch.manual_seed(9)
    model = BitNetLlama(tiny_config(vocab_size=256, max_seq_len=8))
    model.eval()
    name = next(k for k, m in manifest["files"].items()
                if m["role"] == "development" and m["cell"] == "en/web")
    ids = np.fromfile(root / name, dtype="<u2").astype(np.int64)
    x = torch.tensor(np.stack([ids[i * 8:i * 8 + 8] for i in range(2)]))
    y = torch.tensor(np.stack([ids[i * 8 + 1:i * 8 + 9] for i in range(2)]))
    with torch.no_grad():
        logits = model(x)["logits"]
        expected = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1)).item()
    one = score_cells(model, root, batch_size=1)["per_cell"]["en/web"]["loss"]
    two = score_cells(model, root, batch_size=2)["per_cell"]["en/web"]["loss"]
    assert one == pytest.approx(expected, abs=1e-6)
    assert two == pytest.approx(expected, abs=1e-6)
