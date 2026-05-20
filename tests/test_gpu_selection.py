from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

import numpy as np
import torch.nn as nn

from methods.trainers.gpu_selection import (
    cuda_fallback_devices,
    module_inference_device,
    offload_module_to_cpu,
    physical_to_torch_cuda_index,
    query_gpu_free_memory_mib,
    rank_cuda_device_indices,
    resolve_device,
    run_with_cuda_oom_retry,
)
from methods.trainers.isodepth import extract_model_isodepth


class GpuSelectionTests(unittest.TestCase):
    def test_query_gpu_free_memory_mib_parses_csv(self) -> None:
        mock_result = mock.Mock(returncode=0, stdout="0, 1024\n1, 65536\n2, 4096\n")
        with mock.patch("methods.trainers.gpu_selection.subprocess.run", return_value=mock_result):
            free = query_gpu_free_memory_mib()
        self.assertEqual(free, {0: 1024.0, 1: 65536.0, 2: 4096.0})

    def test_rank_prefers_most_free_gpu(self) -> None:
        mock_result = mock.Mock(returncode=0, stdout="0, 1000\n1, 80000\n3, 500\n")
        with (
            mock.patch("methods.trainers.gpu_selection.subprocess.run", return_value=mock_result),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=8),
        ):
            ranked = rank_cuda_device_indices(min_free_mib=0.0)
        self.assertEqual(ranked[0], 1)

    def test_rank_respects_cuda_visible_devices(self) -> None:
        mock_result = mock.Mock(returncode=0, stdout="0, 1000\n1, 80000\n2, 500\n")
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,2"}, clear=False),
            mock.patch("methods.trainers.gpu_selection.subprocess.run", return_value=mock_result),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=2),
        ):
            ranked = rank_cuda_device_indices(min_free_mib=0.0)
        # Physical 1 -> torch 0, physical 2 -> torch 1; physical 1 has more free memory.
        self.assertEqual(ranked, [0, 1])

    def test_physical_to_torch_index_with_visible_devices(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,2,4"}, clear=False):
            self.assertEqual(physical_to_torch_cuda_index(1), 0)
            self.assertEqual(physical_to_torch_cuda_index(4), 2)

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.device_count", return_value=8)
    def test_resolve_device_cuda_picks_best(self, _count: mock.Mock, _avail: mock.Mock) -> None:
        mock_result = mock.Mock(returncode=0, stdout="0, 1000\n1, 80000\n")
        with mock.patch("methods.trainers.gpu_selection.subprocess.run", return_value=mock_result):
            device = resolve_device("cuda")
        self.assertEqual(device, torch.device("cuda:1"))

    def test_cuda_fallback_devices_puts_requested_first(self) -> None:
        mock_result = mock.Mock(returncode=0, stdout="0, 1000\n1, 80000\n2, 70000\n")
        with (
            mock.patch("methods.trainers.gpu_selection.subprocess.run", return_value=mock_result),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=8),
        ):
            devices = cuda_fallback_devices(torch.device("cuda:0"), min_free_mib=0.0)
        self.assertEqual(devices[0], torch.device("cuda:0"))
        self.assertIn(torch.device("cuda:1"), devices)
        self.assertIn(torch.device("cuda:2"), devices)

    def test_run_with_cuda_oom_retry_advances_on_oom(self) -> None:
        calls: list[torch.device] = []

        def fn(device: torch.device) -> str:
            calls.append(device)
            if len(calls) == 1:
                raise torch.cuda.OutOfMemoryError("simulated")
            return "ok"

        mock_result = mock.Mock(returncode=0, stdout="0, 1000\n1, 80000\n")
        with (
            mock.patch("methods.trainers.gpu_selection.subprocess.run", return_value=mock_result),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=8),
            mock.patch("methods.trainers.gpu_selection._clear_cuda_memory"),
        ):
            out = run_with_cuda_oom_retry(fn, torch.device("cuda:0"), min_free_mib=0.0, label="test")
        self.assertEqual(out, "ok")
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])


    def test_offload_module_to_cpu(self) -> None:
        model = nn.Linear(2, 1)
        model = offload_module_to_cpu(model)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_extract_model_isodepth_uses_model_device(self) -> None:
        class _TinyEncoderNet(nn.Module):
            latent_dim = 1

            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Linear(2, 1)

        model = offload_module_to_cpu(_TinyEncoderNet())
        s = np.zeros((3, 2), dtype=np.float32)
        out = extract_model_isodepth(model, s, torch.device("cuda"))
        self.assertEqual(out.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
