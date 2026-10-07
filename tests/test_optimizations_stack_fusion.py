"""Stack fusion operators: CPU reference parity with the eager model math."""

import unittest

import torch

from optimizations.stack_fusion import block_boundary, out_norm_residual, value_mix


def eager_rms_norm(values, gain, eps):
    """RMSNorm module math with autograd through the passed gain tensor."""
    values_fp32 = values.float()
    normalized = values_fp32 * torch.rsqrt(values_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * gain.float()).to(values.dtype)


def eager_out_norm_residual(residual, branch, gain, eps):
    return residual + eager_rms_norm(branch, gain, eps)


def eager_block_boundary(residual, branch, out_gain, in_gain, eps):
    xn = eager_out_norm_residual(residual, branch, out_gain, eps)
    return xn, eager_rms_norm(xn, in_gain, eps)


class TestOutNormResidual(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.residual = torch.randn(3, 5, 32)
        self.branch = torch.randn(3, 5, 32, dtype=torch.bfloat16)
        self.gain = torch.randn(32) * 0.5 + 1.0
        self.eps = 1e-5

    def test_forward_matches_eager(self):
        fused = out_norm_residual(self.residual, self.branch, self.gain, self.eps)
        eager = eager_out_norm_residual(self.residual, self.branch, self.gain, self.eps)
        torch.testing.assert_close(fused, eager, atol=1e-6, rtol=1e-5)

    def test_backward_matches_eager(self):
        residual = self.residual.clone().requires_grad_()
        branch = self.branch.clone().requires_grad_()
        gain = self.gain.clone().requires_grad_()
        out_norm_residual(residual, branch, gain, self.eps).sum().backward()
        eager_residual = self.residual.clone().requires_grad_()
        eager_branch = self.branch.clone().requires_grad_()
        eager_gain = self.gain.clone().requires_grad_()
        eager_out_norm_residual(eager_residual, eager_branch, eager_gain, self.eps).sum().backward()
        torch.testing.assert_close(residual.grad, eager_residual.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(branch.grad, eager_branch.grad, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(gain.grad, eager_gain.grad, atol=1e-4, rtol=1e-3)

    def test_validation(self):
        with self.assertRaises(ValueError):
            out_norm_residual(self.residual, self.branch[:, :, :16], self.gain, self.eps)
        with self.assertRaises(ValueError):
            out_norm_residual(self.residual, self.branch, self.gain[:16], self.eps)
        with self.assertRaises(ValueError):
            out_norm_residual(self.residual, self.branch, self.gain, 0.0)


class TestBlockBoundary(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.residual = torch.randn(2, 7, 32)
        self.branch = torch.randn(2, 7, 32, dtype=torch.bfloat16)
        self.out_gain = torch.randn(32) * 0.5 + 1.0
        self.in_gain = torch.randn(32) * 0.5 + 1.0
        self.eps = 1e-5

    def test_forward_matches_eager(self):
        xn, h = block_boundary(self.residual, self.branch, self.out_gain, self.in_gain, self.eps)
        eager_xn, eager_h = eager_block_boundary(self.residual, self.branch, self.out_gain,
                                                 self.in_gain, self.eps)
        torch.testing.assert_close(xn, eager_xn, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(h, eager_h, atol=1e-6, rtol=1e-5)

    def test_backward_matches_eager(self):
        upstream = torch.randn(2, 7, 32)
        arguments = [t.clone().requires_grad_() for t in
                     (self.residual, self.branch, self.out_gain, self.in_gain)]
        xn, h = block_boundary(*arguments, self.eps)
        ((xn - h) * upstream).sum().backward()
        eager_arguments = [t.clone().requires_grad_() for t in
                           (self.residual, self.branch, self.out_gain, self.in_gain)]
        eager_xn, eager_h = eager_block_boundary(*eager_arguments, self.eps)
        ((eager_xn - eager_h) * upstream).sum().backward()
        for name, fused, eager in zip(("residual", "branch", "out_gain", "in_gain"),
                                      arguments, eager_arguments):
            torch.testing.assert_close(fused.grad, eager.grad, atol=1e-3, rtol=1e-2, msg=lambda m, n=name: f"{n}: {m}")


class TestValueMix(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.v1 = torch.randn(2, 2, 9, 16, dtype=torch.bfloat16)
        self.vn = torch.randn(2, 2, 9, 16, dtype=torch.bfloat16)
        self.lambdas = torch.tensor([0.5, 0.5])

    def test_forward_matches_eager(self):
        fused = value_mix(self.v1, self.vn, self.lambdas)
        eager = self.lambdas[0] * self.v1 + self.lambdas[1] * self.vn
        torch.testing.assert_close(fused, eager, atol=0, rtol=0)

    def test_backward_matches_eager(self):
        upstream = torch.randn_like(self.vn)
        v1, vn, lambdas = (t.clone().requires_grad_() for t in (self.v1, self.vn, self.lambdas))
        (value_mix(v1, vn, lambdas) * upstream).sum().backward()
        e1, en, el = (t.clone().requires_grad_() for t in (self.v1, self.vn, self.lambdas))
        ((el[0] * e1 + el[1] * en) * upstream).sum().backward()
        torch.testing.assert_close(v1.grad, e1.grad, atol=0, rtol=0)
        torch.testing.assert_close(vn.grad, en.grad, atol=0, rtol=0)
        torch.testing.assert_close(lambdas.grad, el.grad, atol=1e-5, rtol=1e-4)

    def test_validation(self):
        with self.assertRaises(ValueError):
            value_mix(self.v1, self.vn[:, :, :, :8], self.lambdas)
        with self.assertRaises(ValueError):
            value_mix(self.v1, self.vn, torch.tensor([0.5]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: kernel parity is a hardware gate")
class TestStackFusionKernelParity(unittest.TestCase):
    def test_out_norm_residual_kernel_matches_reference(self):
        from optimizations import stack_fusion_kernels

        torch.manual_seed(19)
        residual = torch.randn(4, 64, 256, device="cuda")
        branch = torch.randn(4, 64, 256, device="cuda", dtype=torch.bfloat16)
        gain = (torch.randn(256, device="cuda") * 0.5 + 1.0).requires_grad_()
        output, inverse, flat = stack_fusion_kernels.out_norm_residual_forward(
            residual, branch, gain.detach(), 1e-5, torch.float32)
        eager = eager_out_norm_residual(residual, branch, gain.detach(), 1e-5)
        torch.testing.assert_close(output.reshape(branch.shape), eager, atol=1e-2, rtol=1e-2)

    def test_value_mix_kernel_matches_reference(self):
        from optimizations import stack_fusion_kernels

        torch.manual_seed(23)
        v1 = torch.randn(2, 2, 32, 128, device="cuda", dtype=torch.bfloat16)
        vn = torch.randn_like(v1)
        lambdas = torch.tensor([0.3, 0.7], device="cuda")
        fused = stack_fusion_kernels.value_mix_forward(v1, vn, lambdas)
        eager = lambdas[0] * v1 + lambdas[1] * vn
        torch.testing.assert_close(fused, eager, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
