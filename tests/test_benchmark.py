"""Bounded benchmark and saved-result reuse tests with a non-neural stub."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from src import benchmark
from src.config import ExperimentConfig
from src.io import atomic_json, read_json


@pytest.fixture
def benchmark_fixture(tmp_path, monkeypatch):
    events = {"loaded": [], "forward_batch_sizes": [], "verified": [], "synchronized": 0}

    class Images:
        def __len__(self):
            return 5400

        def __getitem__(self, index):
            return torch.full((3, 2, 2), index % 255, dtype=torch.uint8), 0

    class FakeModel:
        def __call__(self, batch):
            events["forward_batch_sizes"].append(len(batch))
            return torch.zeros(len(batch), 10)

    identifiers = [f"validation_{index}" for index in range(5400)]

    def dataset(split):
        assert split == "val", "Benchmark must not request test data"
        return Images()

    data = SimpleNamespace(
        identity="data-identity",
        dataset=dataset,
        sample_ids=lambda split: identifiers if split == "val" else None,
    )
    run = SimpleNamespace(
        path=tmp_path,
        config=ExperimentConfig(),
        identity={"run_uuid": "fixture-run"},
        device=torch.device("cpu"),
        writer_lock=nullcontext,
        verify=lambda current_data: events["verified"].append(current_data.identity),
    )
    selection = {"selected": {name: {"sha256": f"checkpoint-{name}"} for name in run.config.fits()}}
    monkeypatch.setattr(benchmark, "require_selection", lambda run, data: selection)

    def load(run, data, name):
        events["loaded"].append(name)
        return FakeModel()

    def synchronize(device):
        assert device.type == "cpu"
        events["synchronized"] += 1

    times = iter(index / 1000 for index in range(10_000))
    monkeypatch.setattr(benchmark, "load_classifier", load)
    monkeypatch.setattr(benchmark, "synchronize", synchronize)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(benchmark, "prepare_images", lambda images, data, device: images.float())
    monkeypatch.setattr(benchmark, "memory_snapshot", lambda device: {"process_rss_mib": 100.0})
    return run, data, events, selection


def test_only_named_model_validation_requests_and_bounded_measurements(benchmark_fixture):
    run, data, events, _ = benchmark_fixture
    result = benchmark.benchmark_fit(run, data, "middle")
    assert events["loaded"] == ["middle"]
    assert events["forward_batch_sizes"] == [1] * 35 + [64] * 35
    assert events["synchronized"] == 2 + 2 * 60
    assert list((run.path / "benchmarks").iterdir()) == [run.path / "benchmarks" / "middle"]
    assert [item["batch_size"] for item in result["results"]] == [1, 64]
    for item in result["results"]:
        assert item["requests"] == 30
        assert item["p50_ms"] == pytest.approx(1.0)
        assert item["p95_ms"] == pytest.approx(1.0)
        assert item["images_per_second"] == pytest.approx(item["batch_size"] * 1000)
    requests = read_json(run.path / "benchmarks" / "middle" / "requests.json")
    assert len(requests) == 60
    assert requests[0]["sample_ids"] == ["validation_0"]
    assert requests[30]["sample_ids"] == [f"validation_{i}" for i in range(64)]
    assert requests[-1]["sample_ids"] == [f"validation_{i}" for i in range(1856, 1920)]


def test_compatible_reuse_never_loads_or_calls_model(benchmark_fixture, monkeypatch):
    run, data, events, _ = benchmark_fixture
    first = benchmark.benchmark_fit(run, data, "middle")
    count = len(events["forward_batch_sizes"])
    monkeypatch.setattr(
        benchmark, "load_classifier", lambda *args: pytest.fail("Model loaded on reuse")
    )
    second = benchmark.benchmark_fit(run, data, "middle")
    assert second == first
    assert len(events["forward_batch_sizes"]) == count


def test_edited_summary_cannot_claim_different_latency(benchmark_fixture):
    run, data, _, _ = benchmark_fixture
    benchmark.benchmark_fit(run, data, "middle")
    path = run.path / "benchmarks" / "middle" / "summary.json"
    saved = read_json(path)
    saved["results"][0]["p50_ms"] = 999
    atomic_json(path, saved)
    with pytest.raises(RuntimeError, match="summary does not match"):
        benchmark.benchmark_fit(run, data, "middle")


@pytest.mark.parametrize("filename", ["latencies.csv", "requests.json"])
def test_changed_raw_artifact_blocks_reuse(benchmark_fixture, filename, monkeypatch):
    run, data, _, _ = benchmark_fixture
    benchmark.benchmark_fit(run, data, "middle")
    path = run.path / "benchmarks" / "middle" / filename
    path.write_text(path.read_text() + "\ncorruption")
    monkeypatch.setattr(
        benchmark, "load_classifier", lambda *args: pytest.fail("Corruption retriggered model")
    )
    with pytest.raises(RuntimeError, match="Corrupt or incompatible"):
        benchmark.benchmark_fit(run, data, "middle")


def test_foreign_checkpoint_blocks_saved_reuse(benchmark_fixture, monkeypatch):
    run, data, _, selection = benchmark_fixture
    benchmark.benchmark_fit(run, data, "middle")
    selection["selected"]["middle"]["sha256"] = "different-checkpoint"
    monkeypatch.setattr(
        benchmark, "load_classifier", lambda *args: pytest.fail("Foreign result retriggered model")
    )
    with pytest.raises(RuntimeError, match="Corrupt or incompatible"):
        benchmark.benchmark_fit(run, data, "middle")


def test_foreign_identity_blocks_saved_reuse(benchmark_fixture):
    run, data, _, _ = benchmark_fixture
    benchmark.benchmark_fit(run, data, "middle")
    path = run.path / "benchmarks" / "middle" / "summary.json"
    saved = read_json(path)
    saved["identity"]["data"] = "different-data"
    atomic_json(path, saved)
    with pytest.raises(RuntimeError, match="Corrupt or incompatible"):
        benchmark.benchmark_fit(run, data, "middle")


def test_unfrozen_comparison_cannot_benchmark(benchmark_fixture, monkeypatch):
    run, data, events, _ = benchmark_fixture

    def blocked(*args):
        raise RuntimeError("Selection has not been frozen")

    monkeypatch.setattr(benchmark, "require_selection", blocked)
    with pytest.raises(RuntimeError, match="not been frozen"):
        benchmark.benchmark_fit(run, data, "middle")
    assert events["loaded"] == []
    assert not (run.path / "benchmarks").exists()
