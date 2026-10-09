"""OLS losses and encoder gradients must not depend on target storage layout."""
import unittest
import torch
from methods.trainers.isodepth import closed_form_gaussian_mse_torch, closed_form_gaussian_decode_torch


class OlsTargetLayoutTests(unittest.TestCase):
    def test_losses_and_gradients_match_at_small_and_constant_latents(self):
        torch.manual_seed(93)
        for dtype in (torch.float32, torch.float64):
            y = torch.randn(19, 7, dtype=dtype)
            base = torch.randn(3, 19, 1, dtype=dtype)
            for decoder in ('linear', 'quadratic'):
                for scale in (1., 1e-3, 1e-4, 1e-8, 0.):
                    for masked in (False, True):
                        with self.subTest(dtype=dtype, decoder=decoder, scale=scale, masked=masked):
                            mask = None
                            if masked:
                                mask = torch.ones(3, 19, 1, dtype=dtype)
                                mask[:, ::3] = 0
                            z = (base * scale + .25).requires_grad_()
                            shared = closed_form_gaussian_mse_torch(z, y, decoder, loss_mask=mask)
                            gs, = torch.autograd.grad(shared.sum(), z)
                            zb = z.detach().clone().requires_grad_()
                            batched = closed_form_gaussian_mse_torch(
                                zb, y.repeat(3, 1, 1), decoder, loss_mask=mask)
                            gb, = torch.autograd.grad(batched.sum(), zb)
                            self.assertTrue(torch.isfinite(gs).all())
                            self.assertTrue(torch.isfinite(gb).all())
                            tol = 2e-5 if dtype == torch.float32 else 1e-10
                            torch.testing.assert_close(shared, batched, rtol=tol, atol=tol)
                            # Normalize by gradient magnitude to handle tiny latents.
                            norm = max(float(gs.abs().max()), float(gb.abs().max()), 1.)
                            torch.testing.assert_close(gs / norm, gb / norm, rtol=tol, atol=tol)
                            # Explicit residual check for the linear decoder used by these runs.
                            if decoder == "linear":
                                pred = closed_form_gaussian_decode_torch(z, y, decoder, loss_mask=mask)
                                residual = (pred - y).square()
                                explicit = residual.mean((1, 2)) if mask is None else (
                                    (residual * mask).sum((1, 2)) / (mask.sum((1, 2)) * y.shape[1]))
                                torch.testing.assert_close(shared, explicit, rtol=tol, atol=tol)



if __name__ == '__main__':
    unittest.main()
