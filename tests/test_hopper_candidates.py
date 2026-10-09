import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from terminus.model import BitNetLlama, RMSNorm
from terminus.quantization import deepgemm_bitlinear, quantize_fp8_directions, dequantize_activation_fp8_e4m3
from tests.helpers import tiny_config
from tests.test_fp8 import emulated_fp8_gemm, error_ratio


class CandidatePolicyTests(unittest.TestCase):
    def test_candidates_keep_parameters_and_checkpoint_schema(self):
        base = tiny_config(fuse_projections=True, fused_pointwise=True)
        old, new = BitNetLlama(base), BitNetLlama(replace(base, fused_mlp=True))
        self.assertEqual(old.count_parameters(), new.count_parameters())
        new.load_state_dict(old.state_dict(), strict=True)
        for policy in ({"fp8_wgrad_splits": 4}, {"fp8_wgrad_splits": True}, {"fp8_wgrad_splits": 3},
                       {"fused_mlp": True, "fuse_projections": False},
                       {"fused_mlp": True, "ffn_activation": "relu2"},
                       {"local_attention_backend": "cudnn"},
                       {"local_attention_backend": "fa3", "attention_backend": "sdpa_reference", "dropout": 0.1},
                       {"fp8_wgrad_recipe": "2d2d"},
                       {"fp8_wgrad_recipe": "1d2d_bf16"},  # requires deepgemm+full; tiny_config is torch backend
                       {"fused_fp8_producers": True},  # requires fused_pointwise + deepgemm
                       {"loss_backend": "triton_fp8_quack"}):  # requires the FP8 activation recipe
            with self.assertRaises(ValueError):
                replace(base, **policy)

    def test_local_attention_backend_dispatch(self):
        base = tiny_config()  # 2 layers, global_attention_every=2: layer 0 local, layer 1 global
        seen = []

        def recording_attention(query, key, value, *, backend, **kwargs):
            seen.append(backend)
            return torch.zeros_like(query)

        for override, expected in (("", ["sdpa_reference", "sdpa_reference"]),
                                   ("flex", ["flex", "sdpa_reference"])):
            seen.clear()
            model = BitNetLlama(replace(base, local_attention_backend=override))
            ids = torch.randint(0, base.vocab_size, (2, base.max_seq_len))
            with patch("terminus.model.attention", recording_attention):
                model(ids)
            self.assertEqual(seen, expected)

    def test_component_physics_matches_total_and_update_budget(self):
        from terminus.roofline import training_roofline
        from terminus.train import TrainConfig

        for checkpointed in (False, True):
            config = tiny_config(gradient_checkpointing=checkpointed, loss_backend="streaming")
            result = training_roofline(config, TrainConfig(), batch_size=4, measured_tokens_s=45000)
            expected = result["ideal_compute_us_per_token"] * result["tokens_per_update"] / 1000
            self.assertAlmostEqual(sum(result["ideal_component_compute_ms_per_update"].values()), expected)
            self.assertAlmostEqual(result["required_time_reduction_fraction"], 1 - 45000 / 67000)


@unittest.skipUnless(torch.cuda.is_available(), "native CUDA validation required")
class HopperCandidateTests(unittest.TestCase):
    def test_split_quantizer_matches_existing_codes_and_scales_exactly(self):
        for rows in (1, 7, 257):
            x = torch.randn(rows, 768, device="cuda", dtype=torch.bfloat16)[:, ::2]
            for rowwise, transposed in ((True, True), (False, True), (True, False)):
                for eps in (0., 1e-5):
                    actual = quantize_fp8_directions(x, rowwise=rowwise, transposed=transposed, eps=eps, backend="triton_split")
                    expected = quantize_fp8_directions(x, rowwise=rowwise, transposed=transposed, eps=eps, backend="triton")
                    for a, b in zip(actual, expected):
                        if a is not None:
                            self.assertTrue(torch.equal(a[0].float(), b[0].float()))
                            self.assertTrue(torch.equal(a[1], b[1]))
                    if transposed and rows % 128:
                        self.assertEqual(torch.count_nonzero(actual[1][0][:, rows:]).item(), 0)

    def test_fused_dual_quantizer_matches_split_codes_and_scales_exactly(self):
        cases = [torch.randn(rows, 768, device="cuda", dtype=torch.bfloat16)[:, ::2] for rows in (1, 7, 257)]
        x = torch.randn(384, 768, device="cuda", dtype=torch.bfloat16)
        x[5:90, 128:256] = 0  # all-zero rowwise and columnwise groups
        x[200, 300] = 1e30
        cases.append(x[:, ::2])
        for case in cases:
            for eps in (0., 1e-5):
                actual = quantize_fp8_directions(case, transposed=True, eps=eps, backend="triton_fused")
                expected = quantize_fp8_directions(case, transposed=True, eps=eps, backend="triton_split")
                for a, b in zip(actual, expected):
                    self.assertTrue(torch.equal(a[0].float(), b[0].float()))
                    self.assertTrue(torch.equal(a[1], b[1]))

    def test_fp8_head_loss_and_gradients_close_to_bf16(self):
        from terminus.loss import streaming_cross_entropy_sum

        torch.manual_seed(7)
        hidden = torch.randn(2, 2048, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = (torch.randn(8192, 768, device="cuda") * 0.02).requires_grad_(True)
        targets = torch.randint(8192, (2, 2048), device="cuda")
        targets[0, :13] = -100  # ignored rows must flow through both paths
        for chunk in (512, 4096):
            reference = streaming_cross_entropy_sum(hidden, weight, targets, chunk, backend="triton")
            actual = streaming_cross_entropy_sum(hidden, weight, targets, chunk, backend="triton_fp8")
            self.assertLess(error_ratio(actual, reference), 2e-3)
            ga = torch.autograd.grad(actual, (hidden, weight), torch.ones((), device="cuda"), retain_graph=True)
            gb = torch.autograd.grad(reference, (hidden, weight), torch.ones((), device="cuda"))
            for a, b in zip(ga, gb):
                self.assertLess(error_ratio(a, b), 3e-2)
        # 1D2D head wgrad: same forward/dgrad path; only the weight gradient changes.
        one = streaming_cross_entropy_sum(hidden, weight, targets, 4096, backend="triton_fp8", wgrad_recipe="1d1d")
        two = streaming_cross_entropy_sum(hidden, weight, targets, 4096, backend="triton_fp8", wgrad_recipe="1d2d_bf16")
        self.assertTrue(torch.equal(one, two))
        g1 = torch.autograd.grad(one, (hidden, weight), torch.ones((), device="cuda"), retain_graph=True)
        g2 = torch.autograd.grad(two, (hidden, weight), torch.ones((), device="cuda"))
        self.assertTrue(torch.equal(g1[0], g2[0]))
        self.assertLess(error_ratio(g2[1], g1[1]), 5e-2)  # same bound as the block-linear 1D2D gate
        try:
            import quack.cross_entropy  # noqa: F401
        except ImportError:
            self.skipTest("quack-kernels not installed")
        # QuACK CE: same math, FP32 reduction-order/approx-reciprocal differences only.
        three = streaming_cross_entropy_sum(hidden, weight, targets, 4096, backend="triton_fp8_quack")
        self.assertLess(error_ratio(three, one), 1e-5)
        g3 = torch.autograd.grad(three, (hidden, weight), torch.ones((), device="cuda"))
        for a, b in zip(g3, g1):
            self.assertLess(error_ratio(a, b), 1e-3)

    def test_wgrad_1d2d_bf16_branch(self):
        from terminus.fp8_kernels import quantize_blockwise_t
        from terminus.model import BitLinear

        torch.manual_seed(31)
        x = torch.randn(3, 7, 256, device="cuda", dtype=torch.bfloat16)  # 21 rows: ragged
        codes, scales = quantize_blockwise_t(x.reshape(21, 256))
        self.assertEqual(codes.shape, (256, 128))
        self.assertEqual(scales.shape, (2, 1))
        self.assertEqual(torch.count_nonzero(codes[:, 21:]).item(), 0)
        rebuilt = codes.float()[:, :21].T * scales.repeat_interleave(128, 0).T
        self.assertLess(error_ratio(rebuilt, x.reshape(21, 256).float()), 0.2)

        def build(recipe):
            return BitLinear(256, 512, eps=1e-6, quantize_activations=True, weight_scale_layout="blockwise",
                             linear_backend="deepgemm", fp8_backward="full", fp8_quantizer="triton_split",
                             fp8_cache_weights=True, weight_mode="dense", fp8_wgrad_recipe=recipe).cuda()

        reference, candidate = build("1d1d"), build("1d2d_bf16")
        candidate.load_state_dict(reference.state_dict())
        xa = x.clone().requires_grad_(True)
        xb = x.clone().requires_grad_(True)
        dy = torch.randn(3, 7, 512, device="cuda", dtype=torch.bfloat16)
        out_a = reference(xa)
        out_b = candidate(xb)
        self.assertTrue(torch.equal(out_a, out_b))  # forward path unchanged
        ga = torch.autograd.grad(out_a, (xa, reference.weight), dy)
        gb = torch.autograd.grad(out_b, (xb, candidate.weight), dy)
        self.assertTrue(torch.equal(ga[0], gb[0]))  # dgrad path unchanged
        self.assertLess(error_ratio(gb[1], ga[1]), 5e-2)  # wgrad: blockwise scales + bf16 output
        self.assertEqual(gb[1].dtype, torch.float32)

    def test_fused_fp8_producer_emission_bitexact(self):
        from terminus.mlp_kernels import packed_forward
        from terminus.pointwise_kernels import norm_forward

        torch.manual_seed(11)
        x = torch.randn(517, 2304, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2304, device="cuda", dtype=torch.float32)
        out, _, _, fp8 = norm_forward(x, w, 1e-5, emit_fp8=True)
        (ref_codes, ref_scales), _ = quantize_fp8_directions(out, rowwise=True, transposed=False, backend="triton_split")
        self.assertTrue(torch.equal(fp8[0].float(), ref_codes.float()))
        self.assertTrue(torch.equal(fp8[1], ref_scales))

        packed = torch.randn(517, 2 * 5760, device="cuda", dtype=torch.bfloat16)
        gain = torch.tensor(0.71, device="cuda", dtype=torch.float32)
        out2, _, fp82 = packed_forward(packed, gain, 1e-5, emit_fp8=True)
        (ref2c, ref2s), _ = quantize_fp8_directions(out2, rowwise=True, transposed=False, backend="triton_split")
        self.assertTrue(torch.equal(fp82[0].float(), ref2c.float()))
        self.assertTrue(torch.equal(fp82[1], ref2s))

        # End-to-end: sidecar-consuming linear must be bitwise-identical forward.
        from terminus.model import BitLinear

        lin = BitLinear(2304, 512, eps=1e-6, quantize_activations=True, weight_scale_layout="blockwise",
                        linear_backend="deepgemm", fp8_backward="full", fp8_quantizer="triton_split",
                        fp8_cache_weights=True, weight_mode="dense").cuda()
        direct = lin(x)
        (x_codes, x_scales), _ = quantize_fp8_directions(x, rowwise=True, transposed=False, backend="triton_split")
        sided = lin(x, activation_rows=(x_codes, x_scales))
        self.assertTrue(torch.equal(direct, sided))

    def test_norm_grid_large_rows_matches_reference(self):
        for gain in ("scalar", "channel"):
            x = torch.randn(4097, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            dy = torch.randn_like(x)
            fused, reference = (RMSNorm(2048, gain=gain, fused=value).cuda() for value in (True, False))
            actual, expected = fused(x), reference(x)
            ga = torch.autograd.grad(actual, (x, fused.weight), dy)
            gb = torch.autograd.grad(expected, (x, reference.weight), dy)
            for a, b in zip(ga, gb):
                self.assertLess(error_ratio(a, b), .012)

    def test_full_vocabulary_loss_offsets_above_int32(self):
        from terminus.loss_kernels import cross_entropy_inplace

        # 32k tokens x 96k classes exceeds 2^31 elements even in BF16.
        # Exercise both reduction reads and in-place derivative writes.
        if torch.cuda.mem_get_info()[0] < 10_000_000_000:
            self.skipTest("large-index regression needs 10 GB free CUDA memory")
        rows, vocab = 32768, 96000
        logits = torch.zeros(rows, vocab, device="cuda", dtype=torch.bfloat16)
        targets = torch.zeros(rows, device="cuda", dtype=torch.long)
        loss, gradient = cross_entropy_inplace(logits, targets)
        torch.testing.assert_close(loss, torch.tensor(rows * math.log(vocab), device="cuda"), rtol=2e-6, atol=0)
        indices = torch.tensor([0, 22370, rows - 1], device="cuda")
        torch.testing.assert_close(gradient[indices, 0], torch.full((3,), 1 / vocab - 1, device="cuda", dtype=logits.dtype))
        torch.testing.assert_close(gradient[indices, -1], torch.full((3,), 1 / vocab, device="cuda", dtype=logits.dtype))

    def test_packed_qkv_equations_strided_gradients_and_value_only(self):
        from terminus.pointwise import packed_qkv_rope, qk_rope
        from terminus.model import apply_rotary

        for q, kv, dim, normalize_qk in ((16, 4, 128, True), (4, 1, 64, True),
                                          (6, 2, 256, True), (5, 1, 128, False)):
            packed = torch.randn(2, 17, (q + 2 * kv) * dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            cos, sin = (torch.randn(1, 1, 17, dim // 2, device="cuda") for _ in range(2))
            actual = packed_qkv_rope(packed, cos, sin, q, kv, 1e-5, normalize_qk=normalize_qk)
            expected = []
            for index, part in enumerate(packed.split((q * dim, kv * dim, kv * dim), -1)):
                part = part.view(2, 17, q if index == 0 else kv, dim).transpose(1, 2)
                expected.append((qk_rope(part, cos, sin, 1e-5) if normalize_qk
                                 else apply_rotary(part, cos, sin)) if index < 2 else part)
            for a, b in zip(actual, expected):
                self.assertLess(error_ratio(a, b), .006)
            for broadcast in (False, True):
                dy = [torch.ones((), device="cuda", dtype=a.dtype).expand_as(a) if broadcast else torch.randn_like(a) for a in actual]
                ga = torch.autograd.grad(actual, packed, dy, retain_graph=True)[0]
                gb = torch.autograd.grad(expected, packed, dy, retain_graph=True)[0]
                self.assertTrue(ga.is_contiguous())
                self.assertLess(error_ratio(ga, gb), .012)
            ga = torch.autograd.grad(actual[2].sum(), packed, retain_graph=True)[0]
            gb = torch.autograd.grad(expected[2].sum(), packed)[0]
            torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_packed_qkv_learned_gains_match_eager_reference(self):
        from terminus.model import RMSNorm, apply_rotary
        from terminus.pointwise import packed_qkv_rope

        for q, kv, dim in ((16, 4, 128), (4, 1, 64), (6, 2, 256)):
            torch.manual_seed(q * 100 + dim)
            packed = torch.randn(2, 17, (q + 2 * kv) * dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            q_gain = torch.nn.Parameter(torch.linspace(.6, 1.7, dim, device="cuda"))
            k_gain = torch.nn.Parameter(torch.linspace(1.5, .8, dim, device="cuda"))
            cos, sin = (torch.randn(1, 1, 17, dim // 2, device="cuda") for _ in range(2))
            actual = packed_qkv_rope(packed, cos, sin, q, kv, 1e-5, normalize_qk=True,
                                     q_gain=q_gain, k_gain=k_gain)
            reference_gains = (torch.nn.Parameter(q_gain.detach().clone()),
                               torch.nn.Parameter(k_gain.detach().clone()))
            expected = []
            for index, part in enumerate(packed.split((q * dim, kv * dim, kv * dim), -1)):
                part = part.view(2, 17, q if index == 0 else kv, dim).transpose(1, 2)
                if index < 2:
                    gain = reference_gains[index]
                    norm = RMSNorm(dim, gain="channel", fused=True).cuda()
                    norm.weight = gain
                    expected.append(apply_rotary(norm(part), cos, sin))
                else:
                    expected.append(part)
            for a, b in zip(actual, expected):
                self.assertLess(error_ratio(a, b), .006)
            dy = [torch.randn_like(a) for a in actual]
            ga = torch.autograd.grad(actual, [packed, q_gain, k_gain], dy, retain_graph=True)
            gb = torch.autograd.grad(expected, [packed, *reference_gains], dy)
            self.assertTrue(ga[0].is_contiguous())
            self.assertLess(error_ratio(ga[0], gb[0]), .012)
            # Shared channel gains: every head/row contributes; compare tightly.
            torch.testing.assert_close(ga[1], gb[1], rtol=2e-2, atol=2e-4)
            torch.testing.assert_close(ga[2], gb[2], rtol=2e-2, atol=2e-4)

    def test_value_mix_matches_eager_reference(self):
        from terminus.pointwise import value_mix

        torch.manual_seed(811)
        for shape in ((2, 6, 17, 128), (1, 6, 5, 64)):
            v0 = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).requires_grad_()
            v1 = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).requires_grad_()
            lambdas = torch.nn.Parameter(torch.tensor([0.3, -1.2], device="cuda"))
            actual = value_mix(v0, v1, lambdas)
            l2 = torch.nn.Parameter(lambdas.detach().clone())
            expected = l2[0] * v0 + l2[1] * v1
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            dy = torch.randn_like(actual)
            ga = torch.autograd.grad(actual, [v0, v1, lambdas], dy, retain_graph=True)
            gb = torch.autograd.grad(expected, [v0, v1, l2], dy)
            torch.testing.assert_close(ga[0], gb[0], rtol=0, atol=0)
            torch.testing.assert_close(ga[1], gb[1], rtol=0, atol=0)
            torch.testing.assert_close(ga[2], gb[2], rtol=1e-2, atol=1e-3)

    def test_packed_mlp_matches_independent_equations(self):
        from terminus.pointwise import packed_swiglu_norm

        torch.manual_seed(943)
        for dtype in (torch.bfloat16, torch.float32):
            for rows, width in ((1, 128), (17, 384), (257, 5376)):
                for normalized in (False, True):
                    packed = torch.randn(rows, 2 * width, device="cuda", dtype=dtype, requires_grad=True)
                    reference = RMSNorm(width, gain="scalar").cuda()
                    with torch.no_grad():
                        reference.weight.fill_(.71)
                    gain = reference.weight if normalized else None
                    operands = (packed, gain) if normalized else (packed,)
                    actual = packed_swiglu_norm(packed, gain)
                    gate, up = packed.chunk(2, -1)
                    expected = F.silu(gate) * up
                    if normalized:
                        expected = reference(expected)
                    threshold = .012 if dtype == torch.bfloat16 else 2e-5
                    self.assertLess(error_ratio(actual, expected), threshold)
                    for dy in (torch.randn_like(actual), torch.ones((), device="cuda", dtype=dtype).expand_as(actual)):
                        ga = torch.autograd.grad(actual, operands, dy, retain_graph=True)
                        gb = torch.autograd.grad(expected, operands, dy, retain_graph=True)
                        for a, b in zip(ga, gb):
                            torch.testing.assert_close(a, b, rtol=threshold, atol=.02 if dtype == torch.bfloat16 else 1e-4)
                        self.assertTrue(ga[0].is_contiguous())

    def test_packed_mlp_retains_no_product_and_handles_zero(self):
        from terminus.pointwise import packed_swiglu_norm

        packed = torch.zeros(3, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gain = torch.tensor(1., device="cuda", requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            result = packed_swiglu_norm(packed, gain)
        self.assertEqual([t.numel() for t in saved], [packed.numel(), 1, 3])
        result.sum().backward()
        self.assertTrue(torch.equal(packed.grad, torch.zeros_like(packed)))
        self.assertEqual(float(gain.grad), 0.)

    def test_split_wgrad_scales_tails_and_determinism(self):
        from terminus.wgrad_kernels import fp8_wgrad

        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("Hopper required")
        torch.manual_seed(12)
        for tokens in (1, 257, 4096):
            for magnitude in (0., 1e-12, 1.):
                a = (torch.randn(tokens, 256, device="cuda") * magnitude).bfloat16()
                b = torch.randn(tokens, 384, device="cuda", dtype=torch.bfloat16)
                _, ac = quantize_fp8_directions(a, rowwise=False, transposed=True, eps=0, backend="triton")
                _, bc = quantize_fp8_directions(b, rowwise=False, transposed=True, backend="triton")
                # FP64 oracle measures reduction error independently of TF32
                # and the implementation's block-dot accumulation strategy.
                expected = dequantize_activation_fp8_e4m3(*ac).double() @ dequantize_activation_fp8_e4m3(*bc).double().T
                for splits in (1, 2, 4, 8, 16):
                    actual = fp8_wgrad(ac, bc, splits=splits)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertLessEqual(error_ratio(actual, expected), 2e-5)
                    self.assertTrue(torch.equal(actual, fp8_wgrad(ac, bc, splits=splits)))

    def test_split_wgrad_end_to_end_and_requested_gradients(self):
        if torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("Hopper required")
        for need_input, need_weight in ((True, True), (False, True), (True, False)):
            x = torch.randn(257, 256, device="cuda", dtype=torch.bfloat16, requires_grad=need_input)
            w = torch.randn(384, 256, device="cuda", requires_grad=need_weight)
            dy = torch.randn(257, 384, device="cuda", dtype=torch.bfloat16)
            # Isolate the new wgrad kernel from optional DeepGEMM installation.
            with patch("terminus.kernels.deep_gemm_fp8_gemm_nt", side_effect=emulated_fp8_gemm):
                result = deepgemm_bitlinear(x, w, backward_mode="full", quantizer="triton", wgrad_splits=4)
                result.backward(dy)
            self.assertEqual(x.grad is not None, need_input)
            self.assertEqual(w.grad is not None, need_weight)
            if need_weight:
                _, gc = quantize_fp8_directions(dy, rowwise=False, transposed=True, eps=0, backend="triton")
                _, xc = quantize_fp8_directions(x.detach(), rowwise=False, transposed=True, backend="triton")
                expected = emulated_fp8_gemm(gc, xc, rhs_group_size=1, out_dtype=torch.float32)
                self.assertLess(error_ratio(w.grad, expected), 2e-5)
