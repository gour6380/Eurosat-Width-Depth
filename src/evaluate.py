"""Freeze validation selections before evaluating the spatial test partition."""

from __future__ import annotations

import gc
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoints import ArtifactError, fit_identity, load_checkpoint, read_commit
from .io import atomic_csv, atomic_json, canonical_hash, read_json, sha256_file
from .models import analytical_parameter_count, build_model
from .runtime import prepare_images


def _selection_identity(run, data) -> dict:
    return {"run_identity": run.identity, "data_identity": data.identity}


def _completed_commit(run, data, name: str):
    run.config.model(name)
    directory = run.fit_dir(name)
    commit = read_commit(directory, fit_identity(run, data, name, "models"))
    if commit is None or commit["state"] != "completed":
        raise RuntimeError(f"Complete {name} before selecting or evaluating it")
    if commit["completed_epoch"] != run.config.training.max_epochs:
        raise ArtifactError(f"{name} did not complete the configured fixed epoch budget")
    return directory, commit


def _selected_entries(run, data) -> dict:
    entries = {}
    for name in run.config.fits():
        directory, commit = _completed_commit(run, data, name)
        selected = commit["epochs"][commit["best_epoch"] - 1]
        architecture = read_json(directory / "architecture.json")
        expected = {
            "configuration": asdict(run.config.model(name)),
            "classes": run.config.data.classes,
            "parameter_count": analytical_parameter_count(
                run.config.model(name), run.config.data.classes
            ),
            "seed": run.config.training.seed,
            "data_identity": data.identity,
        }
        if any(architecture.get(key) != value for key, value in expected.items()):
            raise ArtifactError(f"Architecture summary does not match the experiment: {directory}")
        entries[name] = {
            "epoch": commit["best_epoch"],
            "sha256": selected["sha256"],
            "file": str((directory / "checkpoints" / selected["file"]).relative_to(run.path)),
            "parameter_count": architecture["parameter_count"],
        }
    return entries


def freeze_selection(run, data) -> dict:
    """Freeze exactly three complete fits before test access."""
    with run.writer_lock():
        run.verify(data)
        path = run.path / "selection.json"
        if path.exists():
            return require_selection(run, data)
        selected = _selected_entries(run, data)
        identity = _selection_identity(run, data)
        result = {
            **identity,
            "identity_sha256": canonical_hash(identity),
            "selected": selected,
            "seed": run.config.training.seed,
            "frozen_at": datetime.now(UTC).isoformat(),
            "rule": "highest validation accuracy; lowest validation CE; earliest epoch",
            "scope": "one seed; all three parameter-matched shapes; no test-based selection",
        }
        atomic_json(path, result)
        return result


def require_selection(run, data) -> dict:
    """Read/verify the test gate without modifying data or fitting a model."""
    run.verify(data)
    path = run.path / "selection.json"
    if not path.exists():
        raise RuntimeError("Test evaluation is blocked until all three main fits are frozen")
    selection = read_json(path)
    identity = _selection_identity(run, data)
    if selection.get("identity_sha256") != canonical_hash(identity):
        raise ArtifactError("Frozen selection belongs to another run or dataset")
    if selection.get("selected") != _selected_entries(run, data):
        raise ArtifactError("Selected checkpoint identities changed after freezing")
    return selection


def load_classifier(run, data, name: str):
    """Load one completed fit's validation-selected weights."""
    run.verify(data)
    directory, commit = _completed_commit(run, data, name)
    state = load_checkpoint(directory, commit, best=True)
    model = build_model(run.config.model(name), classes=run.config.data.classes)
    model.load_state_dict(state["model"])
    model.to(run.device)
    model.eval()
    del state
    return model


@torch.inference_mode()
def predict_images(model, images, data, device) -> dict:
    """Predict uint8 NCHW images. This function does not select a dataset split."""
    model.eval()
    values = prepare_images(images, data, device)
    probabilities = model(values).softmax(-1)
    confidence, labels = probabilities.max(-1)
    return {
        "labels": labels.cpu(),
        "confidence": confidence.cpu(),
        "probabilities": probabilities.cpu(),
        "class_names": list(data.class_names),
    }


def classification_metrics(targets, predictions, classes: int) -> dict:
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if targets.ndim != 1 or predictions.shape != targets.shape or targets.size == 0:
        raise ValueError("Metrics require equally sized nonempty label vectors")
    if (
        not np.issubdtype(targets.dtype, np.integer)
        or not np.issubdtype(predictions.dtype, np.integer)
        or np.any(targets < 0)
        or np.any(targets >= classes)
        or np.any(predictions < 0)
        or np.any(predictions >= classes)
    ):
        raise ValueError("Labels must be integer class IDs from the declared vocabulary")
    confusion = np.bincount(targets * classes + predictions, minlength=classes**2).reshape(
        classes, classes
    )
    tp = confusion.diagonal().astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    # Every test class has support in the protocol. Missing classes are explicit.
    recall = np.divide(tp, support, out=np.full(classes, np.nan), where=support != 0)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted != 0)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.full(classes, np.nan), where=denominator != 0)
    return {
        "accuracy": float(tp.sum() / targets.size),
        "macro_f1": float(f1.mean()) if np.all(np.isfinite(f1)) else None,
        "examples": int(targets.size),
        "confusion": confusion.tolist(),
        "per_class": [
            {
                "class_id": i,
                "support": int(support[i]),
                "predicted_count": int(predicted[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]) if np.isfinite(recall[i]) else None,
                "f1": float(f1[i]) if np.isfinite(f1[i]) else None,
            }
            for i in range(classes)
        ],
        "definitions": {
            "accuracy": "correct images / all test images",
            "macro_f1": "equal-weight mean of class F1; missing true/predicted class is undefined",
            "precision": "zero if a class is never predicted; predicted_count is reported",
        },
    }


def _verify_saved_evaluation(directory: Path, identity: dict) -> dict | None:
    manifest_path = directory / "artifacts.json"
    if not manifest_path.exists():
        return None
    saved = read_json(manifest_path)
    if saved.get("identity_sha256") != canonical_hash(identity):
        raise ArtifactError("Saved evaluation belongs to a different checkpoint or partition")
    if canonical_hash(saved.get("identity")) != saved["identity_sha256"]:
        raise ArtifactError("Saved evaluation identity is corrupt")
    expected_files = {
        "metrics.json",
        "predictions.csv",
        "failures.csv",
        "per_class.csv",
        "confusion.csv",
    }
    if set(saved.get("files", {})) != expected_files:
        raise ArtifactError("Saved evaluation is missing required file checksums")
    for name, digest in saved["files"].items():
        if Path(name).name != name:
            raise ArtifactError("Unexpected saved evaluation path")
        path = directory / name
        if not path.exists() or sha256_file(path) != digest:
            raise ArtifactError(f"Corrupt saved evaluation: {path}")
    return read_json(directory / "metrics.json")


@torch.inference_mode()
def evaluate_fit(run, data, name: str) -> dict:
    """Evaluate one frozen model once, or verify and reuse its saved test results."""
    with run.writer_lock():
        selection = require_selection(run, data)
        selected = selection["selected"][name]
        identity = {
            "run": run.identity,
            "data": data.identity,
            "checkpoint_sha256": selected["sha256"],
            "partition": "test",
            "variant": name,
        }
        directory = run.path / "evaluation" / name
        saved = _verify_saved_evaluation(directory, identity)
        if saved is not None:
            return saved
        # This is the first point where test pixels/labels become accessible.
        dataset = data.dataset("test", allow_test=True)
        ids = data.sample_ids("test")
        if len(ids) != len(dataset):
            raise ArtifactError("Test sample IDs are not aligned with the dataset")
        loader = DataLoader(
            dataset,
            batch_size=run.config.training.validation_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=run.device.type == "cuda",
            generator=torch.Generator().manual_seed(run.config.training.seed),
        )
        model = load_classifier(run, data, name)
        rows, targets, guesses = [], [], []
        loss_sum = torch.zeros((), device=run.device)
        offset = 0
        try:
            for images, labels in tqdm(loader, desc=f"Test: {name}", mininterval=1.0):
                inputs = prepare_images(images, data, run.device)
                labels = labels.to(run.device, non_blocking=run.device.type == "cuda")
                logits = model(inputs)
                loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum")
                confidence, predicted = logits.softmax(-1).max(-1)
                batch = (
                    torch.stack((labels.float(), predicted.float(), confidence), dim=1)
                    .cpu()
                    .tolist()
                )
                for truth, guess, score in batch:
                    if not math.isfinite(score):
                        raise FloatingPointError("Nonfinite test prediction")
                    truth, guess = int(truth), int(guess)
                    targets.append(truth)
                    guesses.append(guess)
                    rows.append(
                        {
                            "sample_id": ids[offset],
                            "target": truth,
                            "prediction": guess,
                            "correct": truth == guess,
                            "confidence": score,
                        }
                    )
                    offset += 1
            loss = float(loss_sum.cpu()) / len(dataset)
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite test loss")
        finally:
            del model
            gc.collect()
        result = classification_metrics(targets, guesses, run.config.data.classes)
        per_class = result.pop("per_class")
        confusion = result.pop("confusion")
        result.update(
            variant=name,
            seed=run.config.training.seed,
            partition="test",
            loss=loss,
            parameter_count=selected["parameter_count"],
            selected_epoch=selected["epoch"],
            identity_sha256=canonical_hash(identity),
            checkpoint_sha256=selected["sha256"],
            confusion_matrix=confusion,
            class_names=list(data.class_names),
        )
        train_counts = data.manifest["splits"]["train"]["class_counts"]
        majority = max(
            range(len(data.class_names)), key=lambda i: train_counts[data.class_names[i]]
        )
        result["majority_baseline_class"] = data.class_names[majority]
        result["majority_baseline_accuracy"] = sum(label == majority for label in targets) / len(
            targets
        )
        for row in per_class:
            row["class_name"] = data.class_names[row["class_id"]]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "metrics.json", result)
        atomic_csv(directory / "predictions.csv", rows)
        atomic_csv(directory / "failures.csv", [row for row in rows if not row["correct"]])
        atomic_csv(directory / "per_class.csv", per_class)
        atomic_csv(
            directory / "confusion.csv",
            [
                {
                    "true_class": data.class_names[i],
                    **{data.class_names[j]: value for j, value in enumerate(row)},
                }
                for i, row in enumerate(confusion)
            ],
        )
        files = (
            "metrics.json",
            "predictions.csv",
            "failures.csv",
            "per_class.csv",
            "confusion.csv",
        )
        atomic_json(
            directory / "artifacts.json",
            {
                "identity": identity,
                "identity_sha256": canonical_hash(identity),
                "files": {file: sha256_file(directory / file) for file in files},
            },
        )
        manifest_path = run.path / "evaluation" / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        manifest[name] = {
            "directory": str(directory.relative_to(run.path)),
            "artifacts_sha256": sha256_file(directory / "artifacts.json"),
        }
        atomic_json(manifest_path, manifest)
        return result


def verify_evaluation(run, data, name: str) -> dict:
    """Saved-only integrity check used by reports; never triggers evaluation."""
    selection = require_selection(run, data)
    identity = {
        "run": run.identity,
        "data": data.identity,
        "checkpoint_sha256": selection["selected"][name]["sha256"],
        "partition": "test",
        "variant": name,
    }
    result = _verify_saved_evaluation(run.path / "evaluation" / name, identity)
    if result is None:
        raise RuntimeError(f"No completed test evaluation is saved for {name}")
    return result


def saved_evaluation(run, name: str) -> dict:
    """Verified saved test results, without loading image arrays or model weights."""
    if run.data_identity is None:
        raise RuntimeError("This run has no bound dataset")
    return verify_evaluation(run, SimpleNamespace(identity=run.data_identity), name)
