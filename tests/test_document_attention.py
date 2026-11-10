import pytest
import torch

from terminus.document_attention import DocumentLayout
from terminus.kernels import attention
from terminus.model import BitNetLlama
from tests.helpers import tiny_config


@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_model_matches_independent_documents_and_gradients(checkpointing):
    config = tiny_config(vocab_size=64, max_seq_len=8, pad_token_id=3, weight_mode="dense", branch_norm="peri",
                         value_residual="learned", qk_norm_mode="learned_rms",
                         gradient_checkpointing=checkpointing)
    torch.manual_seed(7)
    model = BitNetLlama(config)
    independent = BitNetLlama(config)
    independent.load_state_dict(model.state_dict())
    inputs = torch.tensor([[8, 9, 4, 20, 21, 22, 4, 3], [15, 16, 4, 3, 3, 3, 3, 3]])
    targets = torch.tensor([[9, 4, -100, 21, 22, 4, -100, -100], [16, 4, -100, -100, -100, -100, -100, -100]])
    layout = DocumentLayout.from_cpu([[0,1,2,0,1,2,3,0], [0,1,2,0,0,0,0,0]],
                                      [[0,0,0,1,1,1,1,-1], [0,0,0,-1,-1,-1,-1,-1]])
    packed = model(inputs, targets, document_layout=layout)
    loss = 0
    for row, start, end in layout.spans:
        actual = independent(inputs[row:row+1, start:end], targets[row:row+1, start:end])
        torch.testing.assert_close(packed["logits"][row:row+1, start:end], actual["logits"], rtol=2e-5, atol=1e-6)
        loss += actual["loss_sum"]
    torch.testing.assert_close(packed["loss_sum"], loss)
    packed["loss_sum"].backward()
    loss.backward()
    for a, b in zip(model.parameters(), independent.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=2e-6)
    changed = inputs.clone(); changed[0, :3] = torch.tensor([25,26,4])
    torch.testing.assert_close(model(changed, document_layout=layout)["logits"][0,3:7], packed["logits"][0,3:7])



def test_local_window_causal_boundaries_and_shared_pair_segment():
    layout = DocumentLayout.from_cpu([[0,1,2,3,4,0,1,2]], [[0,0,0,0,0,1,1,1]])
    q = torch.zeros(1, 2, 8, 4, requires_grad=True)
    k = torch.zeros(1, 1, 8, 4, requires_grad=True)
    v = torch.arange(8.).reshape(1,1,8,1).expand(1,1,8,4).clone().requires_grad_()
    output = attention(q,k,v,backend="sdpa_reference", local_window=3, document_layout=layout)
    torch.testing.assert_close(output[0,0,:,0], torch.tensor([0.,.5,1.,2.,3.,5.,5.5,6.]))
    # A compound pair is intentionally one segment: its second passage can see its first.
    joined = DocumentLayout.from_cpu([list(range(8))], [[0]*8])
    shared = attention(q,k,v,backend="sdpa_reference",document_layout=joined)
    assert shared[0,0,5,0] == 2.5
    shared.sum().backward()
    assert torch.isfinite(v.grad).all()



@pytest.mark.parametrize("positions,segments", [([[0,1,0]], [[0,-1,1]]),
    ([[0,1,2]], [[0,1,1]]), ([[0,1,0]], [[1,1,-1]])])
def test_bad_layout_rejected(positions, segments):
    with pytest.raises(ValueError):
        DocumentLayout.from_cpu(positions, segments)



@pytest.mark.parametrize("backend", ["rocm_aiter", "rocm_ck"])
@pytest.mark.parametrize("local_window,expected_window", [(512, (511, 0)), (None, (-1, -1))])
def test_native_varlen_adapter_uses_layout_window_gqa_and_backward(monkeypatch, backend, local_window, expected_window):
    import production.amd.attention as amd
    import production.amd.runtime as runtime
    monkeypatch.setattr(runtime, "require_gfx942", lambda device: None)
    calls = []
    def native(q,k,v,**kwargs):
        calls.append((q.shape,k.shape,kwargs))
        return q+(k.sum()+v.sum())*0
    monkeypatch.setattr(amd, "varlen_implementation", lambda backend: native)
    layout = DocumentLayout.from_cpu([[0,1,0,1,2,0]], [[0,0,1,1,1,-1]])
    q = torch.randn(1,4,6,8,dtype=torch.bfloat16,requires_grad=True)
    k,v = [torch.randn(1,2,6,8,dtype=torch.bfloat16,requires_grad=True) for _ in range(2)]
    output = amd.attention(q,k,v,backend=backend,local_window=local_window,document_layout=layout)
    output.sum().backward()
    assert calls[0][0] == (5,4,8) and calls[0][1] == (5,2,8)
    assert calls[0][2]["window_size"] == expected_window
    assert calls[0][2]["causal"] is True and calls[0][2]["dropout_p"] == 0.0
    if backend == "rocm_aiter":
        assert calls[0][2]["how_v3_bf16_cvt"] == 0
    else:
        assert "how_v3_bf16_cvt" not in calls[0][2]
    assert calls[0][2]["cu_seqlens_q"].tolist() == [0,2,5]
    assert not output[:,:,5].any() and not q.grad[:,:,5].any()



def test_attention_stability_sample_excludes_other_docs_padding_and_local_outside_window():
    from terminus.stability import StabilityMonitor

    model = BitNetLlama(tiny_config(max_seq_len=8, local_attention_window=2))
    monitor = StabilityMonitor(model)
    monitor.active = True
    layout = DocumentLayout.from_cpu([[0,1,2,3,4,0,1,0]], [[0,0,0,0,0,1,1,-1]])
    q = torch.ones(1,4,8,8)
    k = torch.ones(1,2,8,8)
    k[:,:,5:] = 10000
    monitor.record_attention(0,q,k,False,document_layout=layout)
    record = monitor.finish()["probes"]["layer_00.attention_sample"]
    assert record["key_positions"] == 2 and record["query_position"] == 4
    assert record["document_mask_applied"] and record["key_start"] == 3
    assert record["logits"]["max_abs"] < 10
    monitor.close()



def test_512_token_local_window_includes_current_token_and_excludes_513th_key():
    positions = [list(range(513))+[0]]
    layout = DocumentLayout.from_cpu(positions, [[0]*513+[1]])
    q = torch.zeros(1,2,514,2)
    k = torch.zeros(1,1,514,2)
    v = torch.arange(514.).reshape(1,1,514,1).expand(1,1,514,2).clone().requires_grad_()
    local = attention(q,k,v,backend="sdpa_reference",local_window=512,document_layout=layout)
    torch.testing.assert_close(local[0,0,511:514,0], torch.tensor([255.5,256.5,513.]))
    local[0,0,512].sum().backward()
    assert not v.grad[:,:,0].any() and v.grad[:,:,1].any()
