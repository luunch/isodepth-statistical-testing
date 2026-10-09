"""Restart sharing must preserve target indexing, gradients, and selected fits."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from data.schemas import TestConfig
from methods.trainers import isodepth as trainer


class RestartTargetStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_slices_across_restart_boundaries(self):
        base = torch.arange(3 * 7 * 5).reshape(3, 7, 5).float()
        cells = torch.tensor([6, 0, 3])
        for repeats in (1, 3, 30):
            expanded = base.repeat_interleave(repeats, dim=0)
            for start, stop in ((0, 2), (1, 3 * repeats - 1)):
                actual = trainer._restart_targets(base, repeats, start, stop, cells)
                expected = expanded[start:stop].index_select(1, cells)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        shared = base[0]
        self.assertIs(trainer._restart_targets(shared, 30), shared)

    def test_cached_norms_preserve_losses_and_gradients(self):
        torch.manual_seed(123)
        for decoder in ('linear', 'quadratic'):
            for repeats in (1, 3):
                base = torch.randn(2, 13, 5)
                targets = trainer._restart_targets(base, repeats)
                z = torch.randn(2 * repeats, 13, 1, requires_grad=True)
                expected = trainer.closed_form_gaussian_mse_torch(z, targets, decoder)
                expected_grad, = torch.autograd.grad(expected.sum(), z)
                actual = trainer.closed_form_gaussian_mse_torch(
                    z, targets, decoder,
                    target_sum_squares=trainer._restart_target_sum_squares(base, repeats),
                )
                actual_grad, = torch.autograd.grad(actual.sum(), z)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_training_matches_explicit_restarts(self):
        rng = np.random.default_rng(91)
        s = rng.normal(size=(2, 17, 2)).astype(np.float32)
        a = rng.normal(size=(2, 17, 5)).astype(np.float32)
        mask = (rng.random((2, 17, 1)) > .2).astype(np.float32)
        for decoder in ('linear', 'quadratic', 'nn'):
            for batch_size in (None, 6):
                for masked in (False, True):
                    with self.subTest(decoder=decoder, batch_size=batch_size, masked=masked):
                        cfg = TestConfig(
                            metric='nll_gaussian_mse', decoder=decoder,
                            n_perms=1, n_reruns=3, epochs=3, patience=0,
                            sgd_batch_size=batch_size, device='cpu', verbose=False, seed=19,
                        )
                        fit_mask = mask if masked else None
                        repeat_inputs = trainer._repeat_batched_inputs
                        def checked_repeat(values, repeats):
                            if values is not None:
                                self.assertNotEqual(values.shape, a.shape,
                                                    'Expression must not be repeated on CPU')
                            return repeat_inputs(values, repeats)
                        finalize = trainer._finalize_batched_parallel_training
                        def checked_finalize(*args, **kwargs):
                            self.assertEqual(tuple(args[2].shape), a.shape)
                            self.assertEqual(kwargs['target_repeats'], 3)
                            return finalize(*args, **kwargs)
                        with patch.object(trainer, '_repeat_batched_inputs', checked_repeat), \
                             patch.object(trainer, '_finalize_batched_parallel_training', checked_finalize):
                            model, outputs = trainer.train_batched_isodepth_model(
                                s, a[0], cfg, a_batched=a, loss_mask_batched=fit_mask,
                            )
                        reference, ref_outputs = trainer.train_batched_isodepth_model(
                            np.repeat(s, 3, axis=0), a[0], replace(cfg, n_reruns=1, n_perms=5),
                            a_batched=np.repeat(a, 3, axis=0),
                            loss_mask_batched=None if fit_mask is None else np.repeat(fit_mask, 3, axis=0),
                        )
                        meta = model.training_metadata
                        ref_meta = reference.training_metadata
                        np.testing.assert_allclose(meta['train_loss_per_rerun'],
                                                   ref_meta['train_loss_per_rerun'].reshape(2, 3),
                                                   rtol=1e-6, atol=1e-7)
                        np.testing.assert_allclose(meta['loss_history_per_slot'],
                                                   ref_meta['loss_history_per_slot'], rtol=1e-6, atol=1e-7)
                        selected = np.arange(2) * 3 + meta['best_rerun_index_per_model']
                        np.testing.assert_allclose(outputs.model_metrics, ref_outputs.model_metrics[selected],
                                                   rtol=1e-6, atol=1e-7)
                        for key, value in model.state_dict().items():
                            torch.testing.assert_close(value, reference.state_dict()[key][selected],
                                                       rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
