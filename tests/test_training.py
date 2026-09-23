"""Runtime acceptance tests; authored for the owner, not run during handoff."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src import train
from src.checkpoints import (
    ArtifactError,
    commit_epoch,
    epoch_rows,
    fit_identity,
    load_checkpoint,
    read_commit,
    write_timing_receipt,
)
from src.config import ExperimentConfig
from src.io import atomic_json, canonical_hash, filesystem_lock, read_json


class TinyRun:
    """Small in-memory protocol adapter; no dataset download or source mutation."""

    def __init__(self, root, epochs=2):
        self.root = Path(root)
        self.path = Path(root) / "run_001"
        self.path.mkdir(parents=True)
        config = ExperimentConfig()
        self.config = replace(
            config,
            training=replace(
                config.training, max_epochs=epochs, batch_size=4, validation_batch_size=4
            ),
        )
        self.device = torch.device("cpu")
        self.identity = {"run_uuid": self.root.name}
        self.data_identity = "tiny-data-v1"

    def writer_lock(self):
        return filesystem_lock(self.path / ".writer.lock")

    def verify(self, data=None):
        if data is not None and data.identity != self.data_identity:
            raise RuntimeError("foreign data")

    def fit_dir(self, name):
        return self.path / "models" / name / "seed_42"

    def update_fit(self, name, status):
        path = self.path / "fits.json"
        content = read_json(path) if path.exists() else {"models": {}}
        content["models"][name] = status
        atomic_json(path, content)


class TinyData:
    identity = "tiny-data-v1"
    mean = (0.5, 0.5, 0.5)
    std = (0.25, 0.25, 0.25)

    def __init__(self):
        images = torch.arange(8 * 3 * 4 * 4).remainder(256).to(torch.uint8).reshape(8, 3, 4, 4)
        self.examples = TensorDataset(images, torch.arange(8).remainder(2))
        self.calls = []

    def dataset(self, split, allow_test=False):
        self.calls.append(split)
        if split == "test":
            raise AssertionError("Training accessed the test partition")
        return self.examples


@pytest.fixture
def tiny_training(monkeypatch):
    # Dropout exercises RNG recovery while remaining a genuinely tiny unit fixture.
    monkeypatch.setattr(
        train,
        "build_model",
        lambda *a, **k: nn.Sequential(
            nn.Flatten(), nn.Linear(48, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 10)
        ),
    )
    monkeypatch.setattr(train, "architecture_table", lambda config: None)
    return TinyData()


def test_selection_rule_uses_loss_then_earliest():
    selected = {"accuracy": 0.8, "loss": 0.5}
    assert train.checkpoint_is_better(0.81, 9.0, selected)
    assert train.checkpoint_is_better(0.8, 0.4, selected)
    assert not train.checkpoint_is_better(0.8, 0.5, selected)
    assert not train.checkpoint_is_better(0.79, 0.1, selected)
    with pytest.raises(FloatingPointError):
        train.checkpoint_is_better(float("nan"), 0.3, selected)


@pytest.mark.parametrize("name", ["shallow_wide", "middle", "deep_narrow"])
def test_any_shape_trains_first_without_prerequisites(tmp_path, tiny_training, name):
    run = TinyRun(tmp_path, epochs=1)
    result = train.train_fit(run, tiny_training, name)
    assert result["completed_epoch"] == run.config.training.max_epochs == 1
    assert set(read_json(run.path / "fits.json")) == {"models"}
    assert set(read_json(run.path / "fits.json")["models"]) == {name}
    assert not (run.path / "pilots").exists()
    assert set(tiny_training.calls) == {"train", "val"}
    for other in run.config.fits():
        assert run.fit_dir(other).exists() == (other == name)


def test_one_fit_does_not_launch_other_shapes_and_reuses_completed(
    tmp_path, tiny_training, monkeypatch
):
    run = TinyRun(tmp_path)
    result = train.train_fit(run, tiny_training, "middle")
    assert result["completed_epoch"] == 2
    assert not run.fit_dir("shallow_wide").exists()
    assert not run.fit_dir("deep_narrow").exists()
    assert set(tiny_training.calls) == {"train", "val"}
    monkeypatch.setattr(
        train, "_loss_epoch", lambda *a, **k: pytest.fail("Completed fit performed more work")
    )
    assert train.train_fit(run, tiny_training, "middle")["reused"]
    statuses = read_json(run.path / "fits.json")
    assert set(statuses) == {"models"}
    assert set(statuses["models"]) == {"middle"}


@pytest.mark.parametrize("failure_point", ["after_commit", "during_partial_epoch"])
def test_recovery_matches_uninterrupted_weights_optimizer_rng_and_loader(
    tmp_path, tiny_training, monkeypatch, failure_point
):
    uninterrupted = TinyRun(tmp_path / "uninterrupted")
    train.train_fit(uninterrupted, tiny_training, "shallow_wide")
    interrupted = TinyRun(tmp_path / "interrupted")
    real_receipt = train.write_timing_receipt
    real_epoch = train._loss_epoch
    completed_training_calls = 0

    def interrupt_after_commit(directory, commit, row):
        if (
            failure_point == "after_commit"
            and Path(directory) == interrupted.fit_dir("shallow_wide")
            and row["epoch"] == 1
        ):
            raise KeyboardInterrupt("simulate power loss after committed weights")
        return real_receipt(directory, commit, row)

    def interrupt_partial_epoch(model, loader, data, run, optimizer=None):
        nonlocal completed_training_calls
        if optimizer is not None:
            completed_training_calls += 1
            if failure_point == "during_partial_epoch" and completed_training_calls == 2:
                # One update changes optimizer/RNG/weights; this epoch must be discarded.
                real_epoch(model, [next(iter(loader))], data, run, optimizer)
                raise KeyboardInterrupt("simulate interruption after a partial epoch update")
        return real_epoch(model, loader, data, run, optimizer)

    monkeypatch.setattr(train, "write_timing_receipt", interrupt_after_commit)
    monkeypatch.setattr(train, "_loss_epoch", interrupt_partial_epoch)
    with pytest.raises(KeyboardInterrupt):
        train.train_fit(interrupted, tiny_training, "shallow_wide")
    identity = fit_identity(interrupted, tiny_training, "shallow_wide", "models")
    committed = read_commit(interrupted.fit_dir("shallow_wide"), identity)
    assert committed["completed_epoch"] == 1
    if failure_point == "after_commit":
        assert (
            epoch_rows(interrupted.fit_dir("shallow_wide"), committed)[0]["epoch_seconds"] is None
        )
    monkeypatch.setattr(train, "write_timing_receipt", real_receipt)
    monkeypatch.setattr(train, "_loss_epoch", real_epoch)
    train.train_fit(interrupted, tiny_training, "shallow_wide")
    states = []
    for run in (uninterrupted, interrupted):
        directory = run.fit_dir("shallow_wide")
        commit = read_commit(directory, fit_identity(run, tiny_training, "shallow_wide", "models"))
        states.append(load_checkpoint(directory, commit))
    first, second = states
    assert first["completed_epoch"] == second["completed_epoch"] == 2
    assert first["global_step"] == second["global_step"] == 4
    for key in first["model"]:
        torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
    for parameter, values in first["optimizer"]["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(
                value, second["optimizer"]["state"][parameter][key], rtol=0, atol=0
            )
    assert torch.equal(first["loader_generator"], second["loader_generator"])
    assert torch.equal(first["validation_generator"], second["validation_generator"])
    assert torch.equal(first["rng"]["torch"], second["rng"]["torch"])
    assert first["scheduler"] == second["scheduler"]
    rows = epoch_rows(
        interrupted.fit_dir("shallow_wide"),
        read_commit(interrupted.fit_dir("shallow_wide"), identity),
    )
    assert [row["epoch"] for row in rows] == [1, 2]
    if failure_point == "after_commit":
        assert rows[0]["timing_receipt"] == "missing_after_checkpoint_commit"
        assert rows[1]["cumulative_active_seconds"] is None
    else:
        assert all(row["timing_receipt"] == "complete" for row in rows)


def test_checkpoint_foreign_corruption_and_alias_recovery(tmp_path):
    identity = {"run": "A"}
    payload = {
        "completed_epoch": 1,
        "identity_sha256": canonical_hash(identity),
        "model": {"w": torch.ones(1)},
    }
    commit = commit_epoch(
        tmp_path,
        identity,
        None,
        payload,
        {"epoch": 1, "validation_accuracy": 0.5, "validation_loss": 1.0},
        improved=True,
        complete=False,
    )
    (tmp_path / "last.pt").unlink()
    read_commit(tmp_path, identity, repair_aliases=True)
    assert (tmp_path / "last.pt").exists()
    with pytest.raises(ArtifactError, match="identity"):
        read_commit(tmp_path, {"run": "B"})
    path = tmp_path / "checkpoints" / commit["epochs"][0]["file"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ArtifactError, match="corrupt"):
        read_commit(tmp_path, identity)


def test_timing_receipt_cannot_bind_wrong_epoch_or_checkpoint(tmp_path):
    identity = {"run": "A"}
    commit = commit_epoch(
        tmp_path,
        identity,
        None,
        {"completed_epoch": 1, "identity_sha256": canonical_hash(identity)},
        {"epoch": 1, "validation_accuracy": 0.5, "validation_loss": 1.0},
        improved=True,
        complete=True,
    )
    with pytest.raises(ArtifactError):
        write_timing_receipt(tmp_path, commit, {"epoch": 2})
    write_timing_receipt(tmp_path, commit, {"epoch": 1, "epoch_seconds": 4.0})
    receipt = read_json(tmp_path / "receipts" / "epoch_001.json")
    receipt["checkpoint_sha256"] = "foreign"
    atomic_json(tmp_path / "receipts" / "epoch_001.json", receipt)
    with pytest.raises(ArtifactError, match="Foreign"):
        epoch_rows(tmp_path, commit)
