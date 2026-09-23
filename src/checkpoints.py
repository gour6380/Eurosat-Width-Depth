"""Epoch transactions: immutable payloads, an atomic commit, then timing receipts.

Only checkpoints produced by this project and verified against their saved SHA-256
are loaded. PyTorch optimizer/RNG checkpoints are trusted local pickle artifacts;
never use this loader for an untrusted download.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import torch

from .io import atomic_csv, atomic_json, canonical_hash, read_json, sha256_file


class ArtifactError(RuntimeError):
    """A saved artifact is corrupt, foreign, or internally inconsistent."""


def fit_identity(run, data, name: str, stage: str) -> dict:
    return {
        "run": run.identity,
        "data": data.identity,
        "configuration_sha256": canonical_hash(run.config.to_dict()),
        "variant": name,
        "seed": run.config.training.seed,
        "stage": stage,
    }


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _alias(source: Path, destination: Path) -> None:
    """Aliases are conveniences; commit.json, not aliases, defines recovery."""
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_commit(directory: Path, identity: dict, *, repair_aliases: bool = False) -> dict | None:
    directory = Path(directory)
    commit_path = directory / "commit.json"
    if not commit_path.exists():
        # Files written before the first atomic commit are uncommitted work.
        # A public alias without its authority must never initialize a new fit.
        if any((directory / name).exists() for name in ("last.pt", "best.pt")):
            raise ArtifactError(f"Checkpoint aliases have no commit manifest: {directory}")
        return None
    try:
        commit = read_json(commit_path)
        if commit["identity_sha256"] != canonical_hash(identity):
            raise ArtifactError(f"Foreign or incompatible fit identity: {directory}")
        if canonical_hash(commit["identity"]) != commit["identity_sha256"]:
            raise ArtifactError(f"Stored fit identity is corrupt: {directory}")
        entries = commit["epochs"]
        if not entries or [e["epoch"] for e in entries] != list(range(1, len(entries) + 1)):
            raise ArtifactError(f"Non-contiguous committed epochs: {directory}")
        if commit["completed_epoch"] != len(entries):
            raise ArtifactError(f"Epoch counter disagrees with manifest: {directory}")
        if commit["best_epoch"] not in range(1, len(entries) + 1):
            raise ArtifactError(f"Invalid selected epoch: {directory}")
        if commit["state"] not in ("completed", "incomplete"):
            raise ArtifactError(f"Invalid committed state: {directory}")
        for entry in entries:
            metrics = entry["metrics"]
            accuracy, loss = metrics["validation_accuracy"], metrics["validation_loss"]
            if (
                metrics["epoch"] != entry["epoch"]
                or not math.isfinite(accuracy)
                or not 0 <= accuracy <= 1
                or not math.isfinite(loss)
                or loss < 0
            ):
                raise ArtifactError(f"Invalid checkpoint-selection metrics: {directory}")
        selected = max(
            entries,
            key=lambda entry: (
                entry["metrics"]["validation_accuracy"],
                -entry["metrics"]["validation_loss"],
                -entry["epoch"],
            ),
        )
        if selected["epoch"] != commit["best_epoch"]:
            raise ArtifactError(
                f"Selected epoch does not follow the fixed validation rule: {directory}"
            )
        for entry in entries:
            expected_name = f"epoch_{entry['epoch']:03d}.pt"
            if entry["file"] != expected_name:
                raise ArtifactError(f"Unexpected checkpoint path: {directory}")
            path = directory / "checkpoints" / expected_name
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ArtifactError(f"Missing or corrupt committed checkpoint: {path}")
        if repair_aliases:
            for alias, epoch in (("last.pt", len(entries)), ("best.pt", commit["best_epoch"])):
                entry = entries[epoch - 1]
                target = directory / alias
                if not target.exists() or sha256_file(target) != entry["sha256"]:
                    _alias(directory / "checkpoints" / entry["file"], target)
        return commit
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ArtifactError(f"Unreadable fit commit at {commit_path}: {exc}") from exc


def load_checkpoint(directory: Path, commit: dict, *, best: bool = False) -> dict:
    index = commit["best_epoch"] - 1 if best else commit["completed_epoch"] - 1
    entry = commit["epochs"][index]
    path = Path(directory) / "checkpoints" / entry["file"]
    if sha256_file(path) != entry["sha256"]:
        raise ArtifactError(f"Checkpoint changed since verification: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ArtifactError(f"Cannot decode committed checkpoint {path}: {exc}") from exc
    if (
        payload.get("identity_sha256") != commit["identity_sha256"]
        or payload.get("completed_epoch") != entry["epoch"]
    ):
        raise ArtifactError(f"Checkpoint payload identity/epoch mismatch: {path}")
    return payload


def commit_epoch(
    directory: Path,
    identity: dict,
    previous: dict | None,
    payload: dict,
    metrics: dict,
    *,
    improved: bool,
    complete: bool,
) -> dict:
    directory = Path(directory)
    epoch = payload["completed_epoch"]
    previous_epoch = 0 if previous is None else previous["completed_epoch"]
    if epoch != previous_epoch + 1:
        raise ArtifactError("Only the next complete epoch may be committed")
    digest = canonical_hash(identity)
    if previous is not None and previous["identity_sha256"] != digest:
        raise ArtifactError("Previous checkpoint belongs to another fit")
    if payload["identity_sha256"] != digest:
        raise ArtifactError("Checkpoint identity does not match this fit")
    name = f"epoch_{epoch:03d}.pt"
    path = directory / "checkpoints" / name
    _atomic_torch_save(path, payload)
    entry = {"epoch": epoch, "file": name, "sha256": sha256_file(path), "metrics": metrics}
    commit = {
        "identity": identity,
        "identity_sha256": digest,
        "completed_epoch": epoch,
        "best_epoch": epoch if improved else previous["best_epoch"],
        "state": "completed" if complete else "incomplete",
        "epochs": ([] if previous is None else previous["epochs"]) + [entry],
    }
    atomic_json(directory / "commit.json", commit)
    # If interrupted here, read_commit(repair_aliases=True) repairs these safely.
    _alias(path, directory / "last.pt")
    selected = commit["epochs"][commit["best_epoch"] - 1]
    if improved:
        _alias(directory / "checkpoints" / selected["file"], directory / "best.pt")
    return commit


def write_timing_receipt(directory: Path, commit: dict, row: dict) -> None:
    entry = commit["epochs"][-1]
    if row["epoch"] != entry["epoch"]:
        raise ArtifactError("A timing receipt cannot describe a different epoch")
    atomic_json(
        Path(directory) / "receipts" / f"epoch_{row['epoch']:03d}.json",
        {
            "identity_sha256": commit["identity_sha256"],
            "checkpoint_sha256": entry["sha256"],
            "metrics": row,
        },
    )


def epoch_rows(directory: Path, commit: dict) -> list[dict]:
    rows = []
    cumulative = 0.0
    known_cumulative = True
    for entry in commit["epochs"]:
        receipt_path = Path(directory) / "receipts" / f"epoch_{entry['epoch']:03d}.json"
        if receipt_path.exists():
            try:
                receipt = read_json(receipt_path)
                if (
                    receipt["identity_sha256"] != commit["identity_sha256"]
                    or receipt["checkpoint_sha256"] != entry["sha256"]
                    or receipt["metrics"]["epoch"] != entry["epoch"]
                ):
                    raise ArtifactError(f"Foreign timing receipt: {receipt_path}")
                row = dict(receipt["metrics"])
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise ArtifactError(f"Corrupt timing receipt: {receipt_path}") from exc
        else:
            row = dict(entry["metrics"])
            row.update(
                checkpoint_seconds=None,
                logging_seconds=None,
                epoch_seconds=None,
                timing_receipt="missing_after_checkpoint_commit",
            )
        if row.get("epoch_seconds") is None:
            known_cumulative = False
        else:
            cumulative += row["epoch_seconds"]
        row["cumulative_active_seconds"] = cumulative if known_cumulative else None
        rows.append(row)
    return rows


def publish_epoch_tables(directory: Path, commit: dict) -> list[dict]:
    rows = epoch_rows(directory, commit)
    atomic_csv(Path(directory) / "epochs.csv", rows)
    atomic_json(Path(directory) / "epochs.json", rows)
    return rows
