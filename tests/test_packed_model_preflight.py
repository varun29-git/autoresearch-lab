"""Exercise the composition gate and prove it detects cross-document leakage."""
from unittest.mock import patch

import pytest
import torch

from terminus.kernels import attention
from terminus.preflight import model_numerical_preflight
from tests.helpers import tiny_config


@pytest.mark.parametrize("mode", ["dense", "ternary"])
def test_packed_composition_checks_loss_gradients_and_isolation(mode):
    config = tiny_config(value_residual="learned", branch_norm="peri",
                         qk_norm_mode="learned_rms", fuse_projections=True,
                         weight_mode=mode, loss_backend="streaming")
    result = model_numerical_preflight(config, torch.device("cpu"), packed=True)
    assert result["packed_documents"]
    assert result["errors"]["max_parameter_gradient_relative_error"] < .03
    assert result["errors"]["document_isolation_max_absolute_error"] == 0


def test_packed_composition_rejects_attention_that_ignores_document_mask():
    config = tiny_config(value_residual="learned", branch_norm="peri", weight_mode="dense")

    def leaking_attention(*args, **kwargs):
        kwargs.pop("document_layout", None)
        return attention(*args, **kwargs)

    with patch("terminus.model.attention", side_effect=leaking_attention):
        with pytest.raises(AssertionError, match="leaked between documents"):
            model_numerical_preflight(config, torch.device("cpu"), packed=True)
