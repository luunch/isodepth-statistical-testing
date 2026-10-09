"""Independent model batching must not alter optimization or checkpoint selection."""
from __future__ import annotations

from dataclasses import replace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

from data.schemas import CovariateConfig, TestConfig
from methods import architectures
from methods.trainers import isodepth as trainer


class TrainingScaleTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        rng = np.random.default_rng(13)
        self.s = rng.normal(size=(12, 2)).astype(np.float32)
        self.a = rng.normal(size=(12, 4)).astype(np.float32)
        self.cov = np.linspace(-1, 1, 12, dtype=np.float32)
        self.labels = np.arange(12) % 2
        self.config = TestConfig(
            n_perms=2, n_reruns=1, epochs=4, patience=0,
            metric='nll_gaussian_mse', decoder='nn', lr=.003,
            seed=13, device='cpu', verbose=False,
        )

    def _run(self, kind, n_models=1, n_reruns=1, batch_size=0,
             metric='nll_gaussian_mse', decoder='nn', corrupt_final=False,
             cosine=False, patience=0, masked=False):
        """Copy identical initial parameters, never assume equal seeds align slots."""
        config = replace(self.config, n_reruns=n_reruns, sgd_batch_size=batch_size,
                         metric=metric, decoder=decoder, sgd_cosine_lr_decay=cosine,
                         patience=patience)
        a = np.abs(self.a) if metric == 'nll_poisson_mse' else self.a
        class_name = {
            'plain': 'ParallelIsoDepthNet',
            'covariate': 'ParallelIsoDepthWithFixedCovariateNet',
            'celltype': 'ParallelCellTypeIsoDepthNet',
            'celltype_covariate': 'ParallelCellTypeIsoDepthWithFixedCovariateNet',
            'fixed': 'ParallelDecoderOnlyNetFixed',
            'midline': 'HybridMidlineParallelNet',
        }[kind]
        base = getattr(architectures, class_name)
        torch.manual_seed(71)
        if kind == 'plain':
            template = base(1, 4, decoder_type=decoder)
        elif kind in ('covariate', 'fixed'):
            template = base(1, 4, self.cov, decoder_type=decoder)
        elif kind == 'celltype':
            template = base(1, 2, 4, decoder_type=decoder)
        elif kind == 'celltype_covariate':
            template = base(1, 2, 4, self.cov, decoder_type=decoder)
        else:
            # Include permutation encoder parameters even for a fixed-only run.
            template = base(2, 4, slot_split=1, decoder_type=decoder)
            config = replace(config, covariate=CovariateConfig(type='midline'))
        weights = dict(template.named_parameters())
        models = []

        class AlignedModel(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                with torch.no_grad():
                    for name, parameter in self.named_parameters():
                        parameter.copy_(weights[name][:1].expand_as(parameter))
                models.append(self)

        steps = []
        lrs = []
        adam = torch.optim.Adam
        steps_per_epoch = (12 + batch_size - 1) // batch_size if batch_size else 1
        final_step = config.epochs * steps_per_epoch

        class RecordingAdam(adam):
            def step(self, closure=None):
                params = [p for group in self.param_groups for p in group['params']]
                steps.append([None if p.grad is None else p.grad[0].detach().clone() for p in params])
                lrs.append(self.param_groups[0]['lr'])
                result = super().step(closure)
                if corrupt_final and len(steps) == final_step:
                    with torch.no_grad():
                        for p in params:
                            p.add_(100.0)
                return result

        with patch.object(trainer, class_name, AlignedModel), patch.object(trainer.optim, 'Adam', RecordingAdam):
            if kind == 'fixed':
                model, pred = trainer.train_fixed_covariate_model(self.cov, a, config)
            elif kind.startswith('celltype'):
                kwargs = {'fixed_covariate_values': self.cov} if kind.endswith('covariate') else {}
                model, outputs, _ = trainer.train_celltype_parallel_isodepth_model(
                    self.s, a, config, cell_type_labels=self.labels, n_cell_types=2,
                    s_batched=np.repeat(self.s[None], n_models, axis=0), **kwargs,
                )
                pred = outputs.pred_true
            else:
                kwargs = {'fixed_covariate_values': self.cov} if kind == 'covariate' else {}
                if masked:
                    mask = np.ones((n_models, 12, 4), dtype=np.float32)
                    mask[:, -3:] = 0
                    kwargs['loss_mask_batched'] = mask
                model, outputs = trainer.train_batched_isodepth_model(
                    np.repeat(self.s[None], n_models, axis=0), a, config, **kwargs,
                )
                pred = outputs.pred_true
        return model, pred, steps, lrs, models[0]

    def _assert_same(self, reference, candidate):
        _, pred0, gradients0, lrs0, _ = reference
        _, pred1, gradients1, lrs1, _ = candidate
        self.assertEqual(lrs0, lrs1)
        self.assertEqual(len(gradients0), len(gradients1))
        for step0, step1 in zip(gradients0, gradients1):
            self.assertEqual(len(step0), len(step1))
            for grad0, grad1 in zip(step0, step1):
                if grad0 is None:
                    self.assertIsNone(grad1)
                else:
                    torch.testing.assert_close(grad0, grad1, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(pred0, pred1, rtol=2e-5, atol=2e-6)

    def test_model_count_and_reruns_do_not_rescale_gradients(self):
        for kind in ('plain', 'covariate', 'celltype', 'celltype_covariate', 'fixed'):
            for batch in (0, 5):
                with self.subTest(kind=kind, batch=batch):
                    reference = self._run(kind, batch_size=batch)
                    self._assert_same(reference, self._run(kind, n_reruns=3, batch_size=batch))
                    if kind != 'fixed':
                        self._assert_same(reference, self._run(kind, n_models=3, batch_size=batch))

    def test_poisson_ols_and_masked_losses_preserve_scale(self):
        for kwargs in (
            {'metric': 'nll_poisson_mse', 'batch_size': 5},
            {'decoder': 'linear'},
            {'decoder': 'quadratic'},
            {'batch_size': 5, 'masked': True},
        ):
            with self.subTest(**kwargs):
                self._assert_same(self._run('plain', **kwargs), self._run('plain', n_models=3, **kwargs))

    def test_best_minibatch_epoch_is_restored_without_patience(self):
        for kind in ('plain', 'covariate', 'celltype', 'celltype_covariate', 'fixed', 'midline'):
            with self.subTest(kind=kind):
                model, pred, _, _, _ = self._run(kind, n_models=3, n_reruns=2,
                                               batch_size=5, corrupt_final=True)
                self.assertLess(float(np.max(np.abs(pred))), 10)
                self.assertTrue(all(float(p.abs().max()) < 10 for p in model.parameters()))
                metadata = trainer.get_training_metadata(model)
                if metadata.get('loss_history_per_slot') is not None:
                    np.testing.assert_allclose(
                        metadata['loss_history_per_slot'][-1],
                        np.asarray(metadata['train_loss_per_rerun']).reshape(-1),
                        atol=2e-6,
                    )

    def test_full_batch_early_stopping_restores_checkpoint(self):
        for kind in ('plain', 'fixed', 'celltype', 'midline'):
            with self.subTest(kind=kind):
                model, pred, _, _, _ = self._run(kind, n_models=3, patience=20, corrupt_final=True)
                self.assertLess(float(np.max(np.abs(pred))), 10)
                self.assertTrue(all(float(p.abs().max()) < 10 for p in model.parameters()))

    def test_cosine_schedule_is_shared_by_all_trainers(self):
        expected = None
        for kind in ('plain', 'covariate', 'celltype', 'celltype_covariate', 'fixed'):
            with self.subTest(kind=kind):
                _, _, _, lrs, _ = self._run(kind, batch_size=5, cosine=True)
                self.assertLess(lrs[-1], lrs[0])
                if expected is None:
                    expected = lrs
                else:
                    np.testing.assert_allclose(lrs, expected, atol=0, rtol=0)

    def test_stopping_other_slots_does_not_rescale_active_gradient(self):
        # Use tiny gradients where Adam epsilon makes averaging consequential.
        for active in ([1., 1., 1.], [1., 0., 0.]):
            p = torch.nn.Parameter(torch.tensor([1e-5, 2e-5, 3e-5]))
            optimizer = torch.optim.Adam([p], lr=.001)
            for _ in range(4):
                optimizer.zero_grad()
                trainer._independent_model_loss(p.square() * 1e-5, torch.tensor(active)).backward()
                optimizer.step()
            if active[1]:
                reference = p[0].item()
            else:
                self.assertEqual(p[0].item(), reference)

    def test_shared_covariate_buffer_is_not_checkpointed_as_slot_state(self):
        model = architectures.ParallelIsoDepthWithFixedCovariateNet(12, 4, self.cov)
        snapshot = trainer._snapshot_parallel_model_state_on_device(model, 12, torch.device('cpu'))
        self.assertNotIn('covariate_values', snapshot)
        self.assertEqual(set(snapshot), set(dict(model.named_parameters())))

    def test_poisson_phase_does_not_restore_gaussian_checkpoint(self):
        config = replace(self.config, metric='nll_poisson_mse', epochs=4,
                         sgd_batch_size=5, gaussian_pretrain_epochs=2,
                         gaussian_pretrain_freeze_encoder=True, patience=1)
        snapshots = []
        original = trainer._update_parallel_model_snapshot_on_device

        def record(snapshot, model, rows, total):
            snapshots.append((any(p.requires_grad for p in model.encoder.parameters()), tuple(rows)))
            return original(snapshot, model, rows, total)

        # A large positive Poisson loss ensures warm MSE would win incorrectly
        # if the best-loss comparator were not reset at the objective switch.
        original_loss = trainer._compute_reconstruction_loss_per_model

        def offset_poisson(*args, **kwargs):
            value = original_loss(*args, **kwargs)
            return value + 1e6 if kwargs.get('poisson_size_factors') is not None else value

        with patch.object(trainer, '_update_parallel_model_snapshot_on_device', record), \
             patch.object(trainer, '_compute_reconstruction_loss_per_model', offset_poisson):
            trainer.train_batched_isodepth_model(self.s[None], np.abs(self.a), config)
        self.assertTrue(any(not trainable and rows == (0,) for trainable, rows in snapshots))


if __name__ == '__main__':
    main()
