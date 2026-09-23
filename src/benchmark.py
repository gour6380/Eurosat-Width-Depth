"""Optional bounded, synchronized model-only inference benchmark on validation images."""

import csv
import time

import numpy as np
import torch

from .evaluate import load_classifier, require_selection
from .io import atomic_csv, atomic_json, canonical_hash, read_json, sha256_file
from .runtime import memory_snapshot, prepare_images, synchronize


def verify_latency_summary(directory, value, settings):
    """Ensure displayed aggregates still follow their checksummed raw measurements."""
    with (directory / "latencies.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(settings.batch_sizes) * settings.repetitions:
        raise RuntimeError("Corrupt benchmark measurement count")
    if [case["batch_size"] for case in value["results"]] != list(settings.batch_sizes):
        raise RuntimeError("Corrupt benchmark case list")
    for case in value["results"]:
        selected = [row for row in rows if int(row["batch_size"]) == case["batch_size"]]
        if [int(row["request"]) for row in selected] != list(range(settings.repetitions)):
            raise RuntimeError("Corrupt benchmark request sequence")
        times = np.asarray([float(row["seconds"]) for row in selected])
        if not np.isfinite(times).all() or np.any(times <= 0):
            raise RuntimeError("Invalid benchmark latency")
        expected = {
            "p50_ms": float(np.quantile(times, 0.5) * 1000),
            "p95_ms": float(np.quantile(times, 0.95) * 1000),
            "images_per_second": case["batch_size"] * len(times) / times.sum(),
        }
        if case["requests"] != len(times) or any(
            not np.isclose(case[key], number, rtol=1e-10, atol=1e-12)
            for key, number in expected.items()
        ):
            raise RuntimeError("Benchmark summary does not match raw latencies")


def benchmark_fit(run, data, name):
    """Measure only the selected named model. No training/test prediction is launched."""
    run.config.model(name)
    with run.writer_lock():
        run.verify(data)
        selected = require_selection(run, data)["selected"][name]
        identity = {
            "run": run.identity,
            "data": data.identity,
            "configuration_sha256": canonical_hash(run.config.to_dict()),
            "variant": name,
            "seed": run.config.training.seed,
            "stage": "models",
        }
        directory = run.path / "benchmarks" / name
        saved = directory / "summary.json"
        if saved.exists():
            value = read_json(saved)
            if (
                value["identity"] != identity
                or value["checkpoint_sha256"] != selected["sha256"]
                or sha256_file(directory / "latencies.csv") != value["raw_sha256"]
                or sha256_file(directory / "requests.json") != value["requests_sha256"]
            ):
                raise RuntimeError("Corrupt or incompatible saved benchmark")
            verify_latency_summary(directory, value, run.config.benchmark)
            return value
        model = load_classifier(run, data, name)
        dataset = data.dataset("val")
        identifiers = data.sample_ids("val")
        settings = run.config.benchmark
        rows, summaries, requests = [], [], []
        try:
            with torch.inference_mode():
                for size in settings.batch_sizes:
                    if len(dataset) < size * settings.repetitions:
                        raise ValueError(
                            "Validation partition too small for registered request panel"
                        )
                    # Normalize and transfer outside measured model-only regions.
                    warm = prepare_images(
                        torch.stack([dataset[i][0] for i in range(size)]), data, run.device
                    )
                    for _ in range(settings.warmups):
                        model(warm)
                    synchronize(run.device)
                    latencies = []
                    for repetition in range(settings.repetitions):
                        indices = list(range(repetition * size, (repetition + 1) * size))
                        images = torch.stack([dataset[i][0] for i in indices])
                        x = prepare_images(images, data, run.device)
                        synchronize(run.device)
                        start = time.perf_counter()
                        output = model(x)
                        synchronize(run.device)
                        seconds = time.perf_counter() - start
                        latencies.append(seconds)
                        rows.append({"batch_size": size, "request": repetition, "seconds": seconds})
                        requests.append(
                            {
                                "batch_size": size,
                                "request": repetition,
                                "sample_ids": [identifiers[i] for i in indices],
                            }
                        )
                        del output, x
                    summaries.append(
                        {
                            "batch_size": size,
                            "requests": len(latencies),
                            "p50_ms": float(np.quantile(latencies, 0.5) * 1000),
                            "p95_ms": float(np.quantile(latencies, 0.95) * 1000),
                            "images_per_second": size * len(latencies) / sum(latencies),
                            **memory_snapshot(run.device),
                        }
                    )
                    del warm
            directory.mkdir(parents=True, exist_ok=True)
            atomic_csv(directory / "latencies.csv", rows)
            atomic_json(directory / "requests.json", requests)
            value = {
                "identity": identity,
                "checkpoint_sha256": selected["sha256"],
                "results": summaries,
                "raw_sha256": sha256_file(directory / "latencies.csv"),
                "requests_sha256": sha256_file(directory / "requests.json"),
                "boundary": "FP32 model forward only; excludes image load, normalization, transfer and prediction formatting",
                "memory_definition": "Post-case snapshots, not peaks; counters are not additive",
                "limitation": "One bounded block per model, sequential order; not a paired hardware optimization study",
            }
            atomic_json(saved, value)
            return value
        finally:
            del model
