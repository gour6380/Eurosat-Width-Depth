"""Backend, normalization and RNG tests; never require a real accelerator."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import runtime


@pytest.mark.parametrize(
    "mps,cuda,expected", [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")]
)
def test_backend_priority(monkeypatch, mps, cuda, expected):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert runtime.select_device().type == expected


def test_silent_mps_fallback_is_rejected(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="Disable"):
        runtime.select_device()


def test_synchronize_calls_only_selected_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    runtime.synchronize(torch.device("cpu"))
    assert calls == []
    runtime.synchronize(torch.device("mps"))
    runtime.synchronize(torch.device("cuda:0"))
    assert calls == ["mps", "cuda:0"]


def test_normalization_is_fp32_train_statistics_and_does_not_consume_rng():
    data = SimpleNamespace(
        mean=np.array([0.25, 0.5, 0.75], dtype=np.float64),
        std=np.array([0.25, 0.5, 0.25], dtype=np.float64),
    )
    images = torch.full((2, 3, 2, 2), 255, dtype=torch.uint8)
    original = images.clone()
    state = torch.get_rng_state().clone()
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        result = runtime.prepare_images(images, data, torch.device("cpu"), augment=False)
    finally:
        torch.set_default_dtype(default)
    assert result.dtype == torch.float32
    assert torch.equal(images, original)
    assert torch.equal(state, torch.get_rng_state())
    assert all(value.dtype == torch.float32 for value in data._normalization_cache["cpu"])
    expected = torch.tensor([3.0, 1.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1)
    assert torch.allclose(result, expected.expand_as(result))


def test_augmentation_is_batched_and_controlled_by_configuration(monkeypatch):
    data = SimpleNamespace(mean=[0.0] * 3, std=[1.0] * 3)
    images = torch.arange(24, dtype=torch.uint8).view(2, 3, 2, 2)
    calls = []

    def fixed_rand(shape, *, device):
        calls.append(shape)
        return torch.tensor([0.0, 1.0], device=device).view(2, 1, 1, 1)

    monkeypatch.setattr(torch, "rand", fixed_rand)
    settings = SimpleNamespace(horizontal_flip=True, vertical_flip=False)
    result = runtime.prepare_images(images, data, torch.device("cpu"), settings)
    assert calls == [(2, 1, 1, 1)]
    assert torch.equal(result[0], images[0].float().flip(-1) / 255)
    assert torch.equal(result[1], images[1].float() / 255)


def test_rng_round_trip_cpu_and_python_numpy(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime.seed_everything(42)
    saved = runtime.capture_rng()
    expected = random.random(), np.random.random(), torch.rand(4)
    runtime.restore_rng(saved)
    actual = random.random(), np.random.random(), torch.rand(4)
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_mps_memory_snapshots_do_not_fabricate_cuda_metrics(monkeypatch):
    monkeypatch.setattr(
        runtime.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100 * 2**20)),
    )
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 20 * 2**20)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 80 * 2**20)
    snapshot = runtime.memory_snapshot(torch.device("mps"))
    assert snapshot == {"process_rss_mib": 100.0, "mps_live_mib": 20.0, "mps_driver_mib": 80.0}
    assert not any("peak" in key or "cuda" in key for key in snapshot)
