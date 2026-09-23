"""Saved-result reporting contracts; no training or inference is required."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import reporting
from src.config import ExperimentConfig
from src.io import atomic_csv, atomic_json, sha256_file


@pytest.fixture
def run(tmp_path, monkeypatch):
    plt.switch_backend("Agg")
    config = ExperimentConfig(training=replace(ExperimentConfig().training, max_epochs=2))
    value = SimpleNamespace(
        path=tmp_path,
        config=config,
        identity={"config_hash": "fixture"},
        data_identity="fixture-data",
        device="mps",
        verify=lambda *args: None,
        writer_lock=lambda: nullcontext(),
        fit_dir=lambda name: tmp_path / "models" / name / "seed_42",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Reporting must not construct, train, evaluate or benchmark a model")

    monkeypatch.setattr("src.models.build_model", forbidden)
    monkeypatch.setattr("src.train.train_fit", forbidden)
    monkeypatch.setattr("src.evaluate.evaluate_fit", forbidden)
    monkeypatch.setattr("src.evaluate.load_classifier", forbidden)
    yield value
    plt.close("all")


def _epochs(run):
    return pd.DataFrame(
        [
            {
                "variant": name,
                "epoch": epoch,
                "epoch_seconds": 12.0,
                "train_seconds": 8.0,
                "validation_seconds": 3.0,
                "checkpoint_seconds": 0.8,
                "logging_seconds": 0.1,
                "train_loss": 1 / epoch,
                "validation_loss": 1.2 / epoch,
                "train_accuracy": 0.5 + epoch * 0.1,
                "validation_accuracy": 0.4 + epoch * 0.1,
                "train_examples_per_second": 2000.0,
                "process_rss_mib": 600.0,
                "mps_live_mib": 300.0,
                "mps_driver_mib": 500.0,
            }
            for name in run.config.fits()
            for epoch in (1, 2)
        ]
    )


def _records(run):
    return [
        {
            "variant": name,
            "seed": 42,
            "accuracy": 0.7,
            "macro_f1": 0.69,
            "parameter_count": 700000 + index,
            "selected_epoch": 2,
            "examples": 100,
            "loss": 0.8,
            "majority_baseline_accuracy": 0.1,
            "class_names": ["Forest", "River"],
            "confusion_matrix": [[35, 15], [15, 35]],
        }
        for index, name in enumerate(run.config.fits())
    ]


def test_timing_summary_missing_receipts_never_become_complete_zero(run):
    frame = _epochs(run)
    frame.loc[frame.variant == "middle", "epoch_seconds"] = np.nan
    frame.loc[(frame.variant == "deep_narrow") & (frame.epoch == 2), "epoch_seconds"] = np.nan
    table = reporting.timing_summary(run, frame).set_index("variant")
    assert table.loc["shallow_wide", "complete_fit_epoch_seconds"] == 24
    assert pd.isna(table.loc["middle", "known_epoch_seconds"])
    assert table.loc["middle", "missing_timing_epochs"] == 2
    assert table.loc["deep_narrow", "known_epoch_seconds"] == 12
    assert pd.isna(table.loc["deep_narrow", "complete_fit_epoch_seconds"])
    empty = reporting.timing_summary(run, pd.DataFrame())
    assert empty["complete_fit_epoch_seconds"].isna().all()


def test_phase_bars_use_same_complete_epochs_for_each_component(run, monkeypatch):
    frame = _epochs(run)
    row = (frame.variant == "shallow_wide") & (frame.epoch == 2)
    frame.loc[row, "train_seconds"] = 100.0
    frame.loc[row, "checkpoint_seconds"] = np.nan
    monkeypatch.setattr(reporting, "epoch_table", lambda current: frame)
    figure = reporting.plot_epoch_costs(run)
    phase_axis = figure.axes[1]
    # If independent means were mixed, the first training bar would incorrectly be 54 seconds.
    assert phase_axis.patches[0].get_height() == 8.0
    assert any("1/2 epochs" in item.get_text() for item in phase_axis.get_xticklabels())


def test_joint_figures_omit_unavailable_cuda_and_no_seed_uncertainty(run, monkeypatch):
    monkeypatch.setattr(reporting, "epoch_table", lambda current: _epochs(run))
    monkeypatch.setattr(reporting, "_evaluation_records", lambda current: _records(run))
    memory = reporting.plot_memory(run)
    assert len(memory.axes) == 3
    assert all("CUDA" not in axis.get_title() for axis in memory.axes)
    learning = reporting.plot_learning_curves(run)
    assert all(len(axis.lines) == 6 for axis in learning.axes)
    assert all(len(axis.collections) == 0 for axis in learning.axes)
    assert len(reporting.plot_throughput(run).axes[0].lines) == 3
    assert all(len(axis.patches) == 3 for axis in reporting.plot_quality(run).axes)
    assert len(reporting.plot_confusions(run).axes) == 3


def test_runtime_figure_omits_missing_cost_instead_of_plotting_zero(run, monkeypatch):
    monkeypatch.setattr(reporting, "_evaluation_records", lambda current: _records(run))
    frame = _epochs(run)
    frame.loc[frame.variant == "middle", "epoch_seconds"] = np.nan
    monkeypatch.setattr(reporting, "epoch_table", lambda current: frame)
    figure = reporting.plot_accuracy_runtime(run)
    assert len(figure.axes[0].collections) == 2
    assert any("Middle" in text.get_text() for text in figure.texts)


def test_failure_review_verifies_each_model_and_ranks_only_wrong_predictions(run, monkeypatch):
    records = _records(run)
    verified = []

    def saved(current, name):
        verified.append(name)
        return next(record for record in records if record["variant"] == name)

    monkeypatch.setattr("src.evaluate.saved_evaluation", saved)
    for name in run.config.fits():
        atomic_csv(
            run.path / "evaluation" / name / "predictions.csv",
            [
                {"sample_id": "correct", "target": 0, "prediction": 0, "confidence": 0.999},
                {"sample_id": "wrong_b", "target": 0, "prediction": 1, "confidence": 0.8},
                {"sample_id": "wrong_a", "target": 1, "prediction": 0, "confidence": 0.9},
            ],
        )
    result = reporting.failure_table(run, examples_per_model=1)
    assert verified == list(run.config.fits())
    assert result.sample_id.tolist() == ["wrong_a"] * 3
    assert result.true_class.tolist() == ["River"] * 3


def test_report_works_without_optional_benchmarks_or_figures(run, monkeypatch):
    monkeypatch.setattr(reporting, "_evaluation_records", lambda current: _records(run))
    frame = _epochs(run)
    frame["epoch_seconds"] = np.nan
    monkeypatch.setattr(reporting, "epoch_table", lambda current: frame)
    destination = reporting.assemble_report(run)
    content = destination.read_text()
    assert "Not executed. No inference-speed claim is made." in content
    assert "700,000–700,002 parameters" in content
    assert "not recorded" in content
    assert "Shallow / wide, Middle, Deep / narrow" in content  # Retain test-score ties.
    assert "no seed error bars" in content


def test_benchmark_request_identity_is_verified(run, monkeypatch):
    name = run.config.fits()[0]
    directory = run.path / "benchmarks" / name
    atomic_csv(directory / "latencies.csv", [{"seconds": 0.001}])
    atomic_json(directory / "requests.json", [{"sample_ids": ["Forest_1.jpg"]}])
    atomic_json(
        directory / "summary.json",
        {
            "identity": reporting._identity(run, name),
            "checkpoint_sha256": "selected",
            "raw_sha256": sha256_file(directory / "latencies.csv"),
            "requests_sha256": sha256_file(directory / "requests.json"),
            "results": [{"batch_size": 1, "p50_ms": 1.0}],
        },
    )
    monkeypatch.setattr(
        "src.evaluate.require_selection", lambda *args: {"selected": {name: {"sha256": "selected"}}}
    )
    assert len(reporting.benchmark_table(run)) == 1
    atomic_json(directory / "requests.json", [{"sample_ids": ["River_9.jpg"]}])
    with pytest.raises(RuntimeError, match="timings changed"):
        reporting.benchmark_table(run)
