"""Run automatically only when the intended native ROCm hardware is present."""
import pytest
import torch

from production.amd.preflight import document_attention_gate, packed_gate
from production.amd.recipe import model_for


native = bool(torch.version.hip and torch.cuda.is_available() and
              str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).startswith("gfx942"))


@pytest.mark.skipif(not native, reason="requires native gfx942 ROCm/AITER runtime")
def test_native_segmented_attention_and_batched_learned_rope():
    model = model_for()
    device = torch.device("cuda", 0)
    assert document_attention_gate(model, device)["cases"]
    assert packed_gate(model, device)["per_batch_reset_rope"]
