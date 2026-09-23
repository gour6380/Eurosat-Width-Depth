"""One shape per call, lightweight epoch logs, and epoch-boundary recovery."""

from __future__ import annotations

import gc
import logging
import math
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .checkpoints import (
    ArtifactError,
    commit_epoch,
    fit_identity,
    load_checkpoint,
    publish_epoch_tables,
    read_commit,
    write_timing_receipt,
)
from .io import atomic_json, atomic_text, canonical_hash, read_json
from .models import architecture_table, build_model
from .runtime import (
    capture_rng,
    memory_snapshot,
    prepare_images,
    restore_rng,
    seed_everything,
    synchronize,
)


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def checkpoint_is_better(accuracy: float, loss: float, selected: dict | None) -> bool:
    """Maximize validation accuracy; break ties by loss, then earliest epoch."""
    if not math.isfinite(accuracy) or not math.isfinite(loss):
        raise FloatingPointError("Nonfinite checkpoint-selection metric")
    return selected is None or (accuracy, -loss) > (selected["accuracy"], -selected["loss"])


def _loader(dataset, run, generator, *, train: bool):
    t = run.config.training
    return DataLoader(
        dataset,
        batch_size=t.batch_size if train else t.validation_batch_size,
        shuffle=train,
        num_workers=0,
        generator=generator,
        pin_memory=run.device.type == "cuda",
        drop_last=False,
    )


def _loss_epoch(model, loader, data, run, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    # Aggregation stays on the selected device until the epoch finishes.
    loss_sum = torch.zeros((), device=run.device, dtype=torch.float32)
    correct = torch.zeros((), device=run.device, dtype=torch.int64)
    grad_sum = torch.zeros((), device=run.device, dtype=torch.float32)
    examples, batches = 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(
            loader, desc="Train" if training else "Validation", leave=False, mininterval=1.0
        ):
            inputs = prepare_images(
                images, data, run.device, augment=run.config.training if training else False
            )
            labels = labels.to(run.device, non_blocking=run.device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                # This safety check may synchronize; it is not optional telemetry.
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(), run.config.training.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                grad_sum += norm.detach()
            loss_sum += loss.detach() * labels.shape[0]
            correct += (logits.detach().argmax(-1) == labels).sum()
            examples += labels.shape[0]
            batches += 1
    if examples == 0:
        raise ValueError("An empty partition cannot be trained or evaluated")
    values = torch.stack((loss_sum, correct.float(), grad_sum)).cpu().tolist()
    if not all(math.isfinite(x) for x in values):
        raise FloatingPointError("Nonfinite aggregate loss, accuracy, or gradient norm")
    return {
        "loss": values[0] / examples,
        "accuracy": values[1] / examples,
        "examples": examples,
        "batches": batches,
        "gradient_norm": values[2] / batches if training else None,
    }


def _logger(directory: Path):
    logger = logging.getLogger(f"eurosat.{directory}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(directory / "training.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def train_fit(run, data, name: str) -> dict:
    """Initialize, resume, or reuse exactly one independently selected fit."""
    run.config.model(name)  # Fail unknown names before any filesystem changes.
    with run.writer_lock():
        run.verify(data)
        directory = run.fit_dir(name)
        identity = fit_identity(run, data, name, "models")
        commit = read_commit(directory, identity, repair_aliases=True)
        limit = run.config.training.max_epochs
        if commit is not None and commit["state"] == "completed":
            if commit["completed_epoch"] != limit:
                raise ArtifactError("Saved epoch limit does not match the configured limit")
            publish_epoch_tables(directory, commit)
            status = {
                "state": "completed",
                "completed_epoch": commit["completed_epoch"],
                "best_epoch": commit["best_epoch"],
                "reused": True,
            }
            run.update_fit(name, status)
            return status
        if (run.path / "selection.json").exists():
            raise RuntimeError("Selected checkpoints are frozen; new training needs a fresh run")
        # Verification may instantiate CPU models only, never several accelerator models.
        architecture_table(run.config)
        directory.mkdir(parents=True, exist_ok=True)
        logger, handler = _logger(directory)
        writer = None
        model = optimizer = scheduler = None
        attempt = {
            "started_at": _utc(),
            "resume_after_epoch": 0 if commit is None else commit["completed_epoch"],
        }
        attempts_path = directory / "attempts.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []
        attempts.append(attempt)
        atomic_json(attempts_path, attempts)
        try:
            seed_everything(run.config.training.seed)
            model = build_model(run.config.model(name), classes=run.config.data.classes).to(
                run.device
            )
            t = run.config.training
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t.max_epochs, eta_min=t.min_learning_rate
            )
            train_generator = torch.Generator().manual_seed(t.seed)
            valid_generator = torch.Generator().manual_seed(t.seed + 1)
            global_step, selected = 0, None
            if commit is not None:
                state = load_checkpoint(directory, commit)
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                train_generator.set_state(state["loader_generator"])
                valid_generator.set_state(state["validation_generator"])
                restore_rng(state["rng"])
                global_step, selected = state["global_step"], state["selected"]
                del state
            start = 1 if commit is None else commit["completed_epoch"] + 1
            if start > limit:
                raise ArtifactError("Incomplete fit has already exceeded its epoch limit")
            train_loader = _loader(data.dataset("train"), run, train_generator, train=True)
            valid_loader = _loader(data.dataset("val"), run, valid_generator, train=False)
            atomic_json(
                directory / "architecture.json",
                {
                    "configuration": asdict(run.config.model(name)),
                    "classes": run.config.data.classes,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "seed": t.seed,
                    "data_identity": data.identity,
                    "initialization": "fresh independently seeded weights; resume restores this fit's last committed epoch",
                },
            )
            atomic_text(directory / "model.txt", str(model) + "\n")
            writer = SummaryWriter(
                str(run.path / "tensorboard" / "models" / name), purge_step=start, flush_secs=30
            )
            status = {
                "state": "running",
                "completed_epoch": start - 1,
                "resume_after_epoch": attempt["resume_after_epoch"],
                "started_at": attempt["started_at"],
            }
            run.update_fit(name, status)
            logger.info("%s: starting epoch %d of %d", name, start, limit)
            for epoch in range(start, limit + 1):
                synchronize(run.device)
                epoch_started = time.perf_counter()
                row = {
                    "variant": name,
                    "seed": t.seed,
                    "stage": "models",
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                started = time.perf_counter()
                training = _loss_epoch(model, train_loader, data, run, optimizer)
                synchronize(run.device)
                row["train_seconds"] = time.perf_counter() - started

                started = time.perf_counter()
                validation = _loss_epoch(model, valid_loader, data, run)
                synchronize(run.device)
                row["validation_seconds"] = time.perf_counter() - started

                row.update(
                    train_loss=training["loss"],
                    train_accuracy=training["accuracy"],
                    validation_loss=validation["loss"],
                    validation_accuracy=validation["accuracy"],
                    train_examples=training["examples"],
                    validation_examples=validation["examples"],
                    optimizer_updates=training["batches"],
                    gradient_norm=training["gradient_norm"],
                    train_examples_per_second=training["examples"] / row["train_seconds"],
                )
                row.update(memory_snapshot(run.device))
                improved = checkpoint_is_better(
                    validation["accuracy"], validation["loss"], selected
                )
                if improved:
                    selected = {
                        "epoch": epoch,
                        "accuracy": validation["accuracy"],
                        "loss": validation["loss"],
                    }
                global_step += training["batches"]
                scheduler.step()

                started = time.perf_counter()
                payload = {
                    "identity_sha256": canonical_hash(identity),
                    "identity": identity,
                    "architecture": asdict(run.config.model(name)),
                    "completed_epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "selected": selected,
                    "rng": capture_rng(),
                    "loader_generator": train_generator.get_state(),
                    "validation_generator": valid_generator.get_state(),
                }
                commit = commit_epoch(
                    directory,
                    identity,
                    commit,
                    payload,
                    dict(row),
                    improved=improved,
                    complete=epoch == limit,
                )
                del payload
                row["checkpoint_seconds"] = time.perf_counter() - started

                started = time.perf_counter()
                for key, value in row.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        writer.add_scalar(key, value, epoch)
                writer.flush()
                logger.info(
                    "epoch=%d train_loss=%.5f val_loss=%.5f val_accuracy=%.4f train_s=%.2f val_s=%.2f selected_epoch=%d",
                    epoch,
                    row["train_loss"],
                    row["validation_loss"],
                    row["validation_accuracy"],
                    row["train_seconds"],
                    row["validation_seconds"],
                    commit["best_epoch"],
                )
                status = {
                    "state": commit["state"],
                    "completed_epoch": epoch,
                    "best_epoch": commit["best_epoch"],
                    "global_step": global_step,
                    "updated_at": _utc(),
                    "reused": False,
                }
                run.update_fit(name, status)
                row["logging_seconds"] = time.perf_counter() - started
                row["epoch_seconds"] = time.perf_counter() - epoch_started
                row["timing_receipt"] = "complete"
                row["timing_boundary"] = (
                    "through epoch logging; excludes final receipt/table/TensorBoard timing-series publication"
                )
                write_timing_receipt(directory, commit, row)
                completed_rows = publish_epoch_tables(directory, commit)
                for key in ("epoch_seconds", "logging_seconds", "cumulative_active_seconds"):
                    value = completed_rows[-1].get(key)
                    if value is not None:
                        writer.add_scalar(f"timing/{key}", value, epoch)
                writer.flush()
            attempt.update(ended_at=_utc(), state="completed", completed_epoch=limit)
            atomic_json(attempts_path, attempts)
            return status
        except BaseException as exc:
            state = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            attempt.update(ended_at=_utc(), state=state, error=f"{type(exc).__name__}: {exc}")
            atomic_json(attempts_path, attempts)
            atomic_text(directory / "last_failure.txt", traceback.format_exc())
            run.update_fit(
                name,
                {
                    "state": state,
                    "completed_epoch": 0 if commit is None else commit["completed_epoch"],
                    "error": attempt["error"],
                },
            )
            logger.exception("Fit stopped; only committed epochs are recoverable")
            raise
        finally:
            try:
                if writer is not None:
                    writer.close()
            finally:
                logger.removeHandler(handler)
                handler.close()
                del model, optimizer, scheduler
                gc.collect()


def training_status(run) -> pd.DataFrame:
    """Read saved states only. Committed epochs are authoritative after a crash."""
    run.verify()
    rows = []
    fits_path = run.path / "fits.json"
    fit_status = read_json(fits_path) if fits_path.exists() else {}
    for name in run.config.fits():
        directory = run.fit_dir(name)
        identity = {
            "run": run.identity,
            "data": run.data_identity,
            "configuration_sha256": canonical_hash(run.config.to_dict()),
            "variant": name,
            "seed": run.config.training.seed,
            "stage": "models",
        }
        commit = read_commit(directory, identity)
        if commit is not None:
            observed = fit_status.get("models", {}).get(name, {}).get("state", commit["state"])
            state = "completed" if commit["state"] == "completed" else observed
            rows.append(
                {
                    "stage": "models",
                    "variant": name,
                    "state": state,
                    "completed_epoch": commit["completed_epoch"],
                    "best_epoch": commit["best_epoch"],
                }
            )
        else:
            saved = fit_status.get("models", {}).get(name, {})
            rows.append(
                {
                    "stage": "models",
                    "variant": name,
                    "state": saved.get("state", "not_started"),
                    "completed_epoch": 0,
                    "best_epoch": None,
                }
            )
    return pd.DataFrame(rows)
