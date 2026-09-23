"""Metrics and test-partition gates, using synthetic unit fixtures only."""

from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.checkpoints import ArtifactError, commit_epoch, fit_identity
from src.config import ExperimentConfig
from src.evaluate import (
    classification_metrics,
    evaluate_fit,
    freeze_selection,
    require_selection,
    saved_evaluation,
)
from src.io import atomic_json, atomic_text, canonical_hash, filesystem_lock, sha256_file
from src.models import analytical_parameter_count


class EvaluationRun:
    def __init__(self, root):
        self.path = Path(root)
        self.root = self.path
        self.identity = {"run_uuid": "evaluation-unit-fixture"}
        self.data_identity = "fixture-data"
        config = ExperimentConfig()
        self.config = replace(config, training=replace(config.training, max_epochs=1))
        self.device = torch.device("cpu")

    def writer_lock(self):
        return filesystem_lock(self.path / ".writer.lock")

    def fit_dir(self, name):
        return self.path / "models" / name / "seed_42"

    def verify(self, data=None):
        if data is not None and data.identity != self.data_identity:
            raise RuntimeError("foreign data")


def _complete(run, name):
    data = SimpleNamespace(identity=run.data_identity)
    identity = fit_identity(run, data, name, "models")
    directory = run.fit_dir(name)
    commit_epoch(
        directory,
        identity,
        None,
        {"identity_sha256": canonical_hash(identity), "completed_epoch": 1},
        {"epoch": 1, "validation_accuracy": 0.8, "validation_loss": 0.6},
        improved=True,
        complete=True,
    )
    atomic_json(
        directory / "architecture.json",
        {
            "configuration": asdict(run.config.model(name)),
            "classes": run.config.data.classes,
            "parameter_count": analytical_parameter_count(run.config.model(name)),
            "seed": run.config.training.seed,
            "data_identity": data.identity,
        },
    )


def test_accuracy_macro_f1_known_answer():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert result["confusion"] == [[1, 1], [0, 2]]
    assert result["per_class"][1]["support"] == 2


def test_undefined_and_never_predicted_classes_are_explicit():
    result = classification_metrics([0, 1], [0, 0], 3)
    assert result["per_class"][1]["precision"] == 0
    assert result["per_class"][1]["predicted_count"] == 0
    assert result["per_class"][1]["f1"] == 0
    assert result["per_class"][2]["recall"] is None
    assert result["macro_f1"] is None
    with pytest.raises(ValueError):
        classification_metrics([0, 2], [0, 0], 2)
    with pytest.raises(ValueError):
        classification_metrics([0.5], [0], 2)


def test_test_pixels_inaccessible_before_all_three_checkpoints_frozen(tmp_path):
    run = EvaluationRun(tmp_path)
    data = SimpleNamespace(
        identity=run.data_identity,
        dataset=lambda *a, **k: pytest.fail("Test accessed before the gate"),
    )
    _complete(run, "shallow_wide")
    with pytest.raises(RuntimeError, match="blocked"):
        evaluate_fit(run, data, "shallow_wide")
    with pytest.raises(RuntimeError, match="Complete"):
        freeze_selection(run, data)
    assert not (run.path / "selection.json").exists()


def test_freeze_requires_exact_complete_main_comparison_and_is_reused(tmp_path):
    run = EvaluationRun(tmp_path)
    data = SimpleNamespace(identity=run.data_identity)
    for name in run.config.fits():
        _complete(run, name)
    selection = freeze_selection(run, data)
    assert set(selection["selected"]) == set(run.config.fits())
    assert freeze_selection(run, data) == selection
    assert require_selection(run, data) == selection
    checkpoint = run.path / selection["selected"]["middle"]["file"]
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="corrupt"):
        require_selection(run, data)


def test_corrupt_architecture_summary_is_rejected(tmp_path):
    run = EvaluationRun(tmp_path)
    data = SimpleNamespace(identity=run.data_identity)
    for name in run.config.fits():
        _complete(run, name)
    atomic_json(run.fit_dir("middle") / "architecture.json", {"parameter_count": 1})
    with pytest.raises(ArtifactError, match="Architecture summary"):
        freeze_selection(run, data)


def test_saved_evaluation_is_read_only_and_verifies_all_five_artifacts(tmp_path):
    run = EvaluationRun(tmp_path)
    data = SimpleNamespace(identity=run.data_identity)
    for name in run.config.fits():
        _complete(run, name)
    selection = freeze_selection(run, data)
    name = "middle"
    identity = {
        "run": run.identity,
        "data": data.identity,
        "checkpoint_sha256": selection["selected"][name]["sha256"],
        "partition": "test",
        "variant": name,
    }
    directory = run.path / "evaluation" / name
    metrics = {
        "variant": name,
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "class_names": ["a", "b"],
        "confusion_matrix": [[1, 1], [0, 0]],
    }
    atomic_json(directory / "metrics.json", metrics)
    csv_files = ("predictions.csv", "failures.csv", "per_class.csv", "confusion.csv")
    for file in csv_files:
        atomic_text(directory / file, "fixture\n")
    manifest = {
        "identity": identity,
        "identity_sha256": canonical_hash(identity),
        "files": {file: sha256_file(directory / file) for file in ("metrics.json", *csv_files)},
    }
    atomic_json(directory / "artifacts.json", manifest)
    assert saved_evaluation(run, name) == metrics
    (directory / "predictions.csv").write_text("tampered\n")
    with pytest.raises(ArtifactError, match="Corrupt"):
        saved_evaluation(run, name)
    manifest["files"].pop("predictions.csv")
    atomic_json(directory / "artifacts.json", manifest)
    with pytest.raises(ArtifactError, match="required"):
        saved_evaluation(run, name)
