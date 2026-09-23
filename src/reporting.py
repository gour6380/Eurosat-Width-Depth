"""Joint, saved-result figures and reports. These functions never execute a model."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .checkpoints import epoch_rows, read_commit
from .io import atomic_csv, atomic_text, canonical_hash, read_json
from .models import DISPLAY_NAMES

COLORS = {"shallow_wide": "#0072B2", "middle": "#D55E00", "deep_narrow": "#009E73"}


def _identity(run, name, stage="models"):
    return {
        "run": run.identity,
        "data": run.data_identity,
        "configuration_sha256": canonical_hash(run.config.to_dict()),
        "variant": name,
        "seed": run.config.training.seed,
        "stage": stage,
    }


def epoch_table(run):
    run.verify()
    rows = []
    for name in run.config.fits():
        directory = run.fit_dir(name)
        commit = read_commit(directory, _identity(run, name))
        if commit:
            rows.extend(dict(row, variant=name) for row in epoch_rows(directory, commit))
    frame = pd.DataFrame(rows)
    with run.writer_lock():
        atomic_csv(run.path / "tables" / "epochs.csv", rows)
    return frame


def _save(run, fig, name):
    fig.set_facecolor("white")
    directory = run.path / "figures"
    with run.writer_lock():
        directory.mkdir(exist_ok=True)
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
    return fig


def _empty(title):
    fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    ax.text(0.5, 0.5, "Not executed: no saved results yet", ha="center", va="center")
    ax.set_title(title)
    ax.axis("off")
    return fig


def plot_learning_curves(run):
    frame = epoch_table(run)
    if frame.empty:
        return _empty("Learning curves")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for name, group in frame.groupby("variant", sort=False):
        for split, style in [("train", "--"), ("validation", "-")]:
            label = f"{DISPLAY_NAMES[name]} · {split}"
            axes[0].plot(
                group.epoch, group[f"{split}_loss"], style, color=COLORS[name], label=label
            )
            axes[1].plot(group.epoch, 100 * group[f"{split}_accuracy"], style, color=COLORS[name])
    axes[0].set(ylabel="Cross-entropy (lower is better)", title="Loss: train and validation")
    axes[1].set(
        ylabel="Correct images (%) · higher is better", title="Image classification accuracy"
    )
    for ax in axes:
        ax.set_xlabel("Completed epoch")
        ax.grid(alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Three shapes · one seed · dashed training / solid validation")
    return _save(run, fig, "learning_curves")


def plot_epoch_costs(run):
    frame = epoch_table(run)
    if frame.empty:
        return _empty("Epoch duration")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for name, group in frame.groupby("variant", sort=False):
        axes[0].plot(
            group.epoch, group.epoch_seconds, label=DISPLAY_NAMES[name], color=COLORS[name]
        )
    axes[0].set(
        xlabel="Completed epoch", ylabel="Seconds (lower is better)", title="Measured full epochs"
    )
    columns = ["train_seconds", "validation_seconds", "checkpoint_seconds", "logging_seconds"]
    # Compare phase means using the same epochs for every component of a bar.
    # A missing receipt must not turn into zero persistence cost or mix denominators.
    complete = frame.dropna(subset=columns)
    means = complete.groupby("variant")[columns].mean()
    means = means.reindex([n for n in run.config.fits() if n in means.index])
    bottom = np.zeros(len(means))
    for col, label in [
        ("train_seconds", "Training"),
        ("validation_seconds", "Validation"),
        ("checkpoint_seconds", "Checkpoint"),
        ("logging_seconds", "Logging"),
    ]:
        axes[1].bar(np.arange(len(means)), means[col], bottom=bottom, label=label)
        bottom += means[col].fillna(0).to_numpy()
    labels = [
        f"{DISPLAY_NAMES[n]}\n{len(complete[complete.variant == n])}/{len(frame[frame.variant == n])} epochs"
        for n in means.index
    ]
    axes[1].set_xticks(np.arange(len(means)), labels)
    axes[1].set(
        ylabel="Mean seconds over epochs with every phase recorded",
        title="Recorded phase boundaries",
    )
    if means.empty:
        axes[1].text(
            0.5,
            0.5,
            "No epochs have complete phase timings",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
        )
    axes[0].legend(loc="best")
    if not means.empty:
        axes[1].legend(loc="best")
    fig.suptitle("Epoch costs · final receipt publication excluded · missing timings stay missing")
    return _save(run, fig, "epoch_costs")


def plot_throughput(run):
    frame = epoch_table(run)
    if frame.empty:
        return _empty("Training throughput")
    fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
    for name, group in frame.groupby("variant", sort=False):
        ax.plot(
            group.epoch,
            group.train_examples_per_second,
            label=DISPLAY_NAMES[name],
            color=COLORS[name],
        )
    ax.set(
        xlabel="Completed epoch",
        ylabel="Training images / training second (higher is better)",
        title="Training throughput includes loading, augmentation and optimizer updates",
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    return _save(run, fig, "throughput")


def plot_memory(run):
    frame = epoch_table(run)
    available = [
        (col, label)
        for col, label in (
            ("process_rss_mib", "Process RSS"),
            ("mps_live_mib", "MPS live allocation"),
            ("mps_driver_mib", "MPS driver allocation"),
            ("cuda_allocated_mib", "CUDA allocation"),
            ("cuda_reserved_mib", "CUDA reserved"),
        )
        if col in frame and frame[col].notna().any()
    ]
    if not available:
        return _empty("Memory snapshots")
    fig, axes = plt.subplots(
        len(available), 1, figsize=(10, 3.1 * len(available)), layout="constrained", squeeze=False
    )
    for ax, (col, label) in zip(axes[:, 0], available, strict=True):
        for name, group in frame.groupby("variant", sort=False):
            ax.plot(group.epoch, group[col], label=DISPLAY_NAMES[name], color=COLORS[name])
        ax.set(xlabel="Completed epoch", ylabel="MiB snapshot", title=label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Observed memory snapshots · neither additive nor measured peaks")
    return _save(run, fig, "memory")


def _evaluation_records(run):
    # Evaluator verifies freeze identities and saved prediction checksums without model calls.
    from .evaluate import saved_evaluation

    return [saved_evaluation(run, name) for name in run.config.fits()]


def quality_table(run):
    records = _evaluation_records(run)
    frame = pd.DataFrame(
        [
            {k: v for k, v in record.items() if not isinstance(v, (dict, list, tuple))}
            for record in records
        ]
    )
    with run.writer_lock():
        atomic_csv(run.path / "tables" / "quality.csv", frame.to_dict("records"))
    return frame


def failure_table(run, examples_per_model=5):
    """Review each shape's highest-confidence saved mistakes without re-evaluation.

    This deliberate diagnostic sample is not a random estimate of error types.
    Complete saved failures remain in each evaluation directory.
    """
    if type(examples_per_model) is not int or examples_per_model < 1:
        raise ValueError("Choose a positive number of failure examples per model")
    records = _evaluation_records(run)
    rows = []
    for record in records:
        name = record["variant"]
        predictions = pd.read_csv(run.path / "evaluation" / name / "predictions.csv")
        errors = predictions.loc[predictions.target != predictions.prediction]
        errors = errors.sort_values(
            ["confidence", "sample_id"], ascending=[False, True], kind="stable"
        )
        labels = record["class_names"]
        for rank, row in enumerate(errors.head(examples_per_model).itertuples(index=False), 1):
            rows.append(
                {
                    "variant": name,
                    "confidence_rank_within_errors": rank,
                    "sample_id": row.sample_id,
                    "true_class": labels[int(row.target)],
                    "predicted_class": labels[int(row.prediction)],
                    "confidence": row.confidence,
                }
            )
    columns = [
        "variant",
        "confidence_rank_within_errors",
        "sample_id",
        "true_class",
        "predicted_class",
        "confidence",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    with run.writer_lock():
        atomic_csv(run.path / "tables" / "failure_review.csv", rows)
    return frame


def plot_quality(run):
    frame = quality_table(run)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, metric, title in zip(
        axes, ["accuracy", "macro_f1"], ["Test accuracy", "Test macro F1"], strict=True
    ):
        values = frame[metric].to_numpy() * 100
        ax.bar(range(len(frame)), values, color=[COLORS[n] for n in frame.variant])
        ax.set_xticks(range(len(frame)), [DISPLAY_NAMES[n] for n in frame.variant])
        ax.set(ylabel="Percent · higher is better", title=title, ylim=(0, 105))
        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.2f}", ha="center", fontsize=11)
    fig.suptitle("Held-out geographic split · one seed · no seed uncertainty estimate")
    return _save(run, fig, "quality")


def plot_confusions(run):
    records = _evaluation_records(run)
    # Stack large square matrices vertically to keep long class names readable.
    fig, axes = plt.subplots(3, 1, figsize=(10, 24), layout="constrained")
    for ax, rec in zip(axes, records, strict=True):
        matrix = np.array(rec["confusion_matrix"], dtype=float)
        if matrix.shape != (len(rec["class_names"]), len(rec["class_names"])):
            raise ValueError("Saved confusion matrix dimensions disagree with class names")
        support = matrix.sum(axis=1, keepdims=True)
        proportions = np.divide(
            matrix, support, out=np.full_like(matrix, np.nan), where=support > 0
        )
        ax.imshow(proportions, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        labels = rec["class_names"]
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i, j in np.ndindex(matrix.shape):
            text = f"{proportions[i, j]:.0%}" if np.isfinite(proportions[i, j]) else "N/A"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if proportions[i, j] > 0.5 else "black",
            )
        ax.set(title=DISPLAY_NAMES[rec["variant"]], xlabel="Predicted class", ylabel="True class")
    fig.suptitle(
        "Confusion matrices · each row is a true class, percentages divide by its test count"
    )
    return _save(run, fig, "confusion_matrices")


def benchmark_table(run):
    run.verify()
    rows = []
    for name in run.config.fits():
        path = run.path / "benchmarks" / name / "summary.json"
        if path.exists():
            value = read_json(path)
            if value["identity"] != _identity(run, name):
                raise RuntimeError("Foreign benchmark results")
            from .io import sha256_file

            raw = path.parent / "latencies.csv"
            if (
                sha256_file(raw) != value["raw_sha256"]
                or sha256_file(path.parent / "requests.json") != value["requests_sha256"]
            ):
                raise RuntimeError("Benchmark timings changed")
            from .evaluate import require_selection

            selected = require_selection(run, SimpleNamespace(identity=run.data_identity))[
                "selected"
            ][name]
            if value["checkpoint_sha256"] != selected["sha256"]:
                raise RuntimeError("Benchmark belongs to a different selected checkpoint")
            from .benchmark import verify_latency_summary

            verify_latency_summary(path.parent, value, run.config.benchmark)
            rows.extend(dict(row, variant=name) for row in value["results"])
    return pd.DataFrame(rows)


def timing_summary(run, frame=None):
    """Summarize only saved epochs, with explicit coverage for missing receipts."""
    if frame is None:
        frame = epoch_table(run)
    rows = []
    for name in run.config.fits():
        group = frame.loc[frame.variant == name] if not frame.empty else pd.DataFrame()
        durations = pd.to_numeric(group.get("epoch_seconds", pd.Series(dtype=float)))
        known = durations.dropna()
        complete = len(group) == run.config.training.max_epochs and len(known) == len(group)
        rows.append(
            {
                "variant": name,
                "completed_epochs": len(group),
                "timed_epochs": len(known),
                "missing_timing_epochs": len(group) - len(known),
                "mean_recorded_epoch_seconds": known.mean() if len(known) else None,
                "known_epoch_seconds": known.sum() if len(known) else None,
                "complete_fit_epoch_seconds": known.sum() if complete else None,
            }
        )
    return pd.DataFrame(rows)


def plot_accuracy_runtime(run):
    quality = quality_table(run)
    epochs = epoch_table(run)
    costs = timing_summary(run, epochs).set_index("variant")
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    omitted = []
    for name in run.config.fits():
        q = quality.loc[quality.variant == name, "accuracy"].iloc[0]
        seconds = costs.loc[name, "complete_fit_epoch_seconds"]
        if pd.isna(seconds):
            omitted.append(DISPLAY_NAMES[name])
            continue
        hours = seconds / 3600
        ax.scatter(hours, 100 * q, s=90, color=COLORS[name], label=DISPLAY_NAMES[name])
    ax.set(
        xlabel="Active full-fit epoch time (hours; lower is faster)",
        ylabel="Test accuracy (%) · higher is better",
        title="Quality versus measured training cost",
    )
    if len(omitted) < len(run.config.fits()):
        ax.legend(loc="best")
    if omitted:
        fig.supxlabel(
            "Omitted because full-fit timing is incomplete: " + ", ".join(omitted), fontsize=9
        )
    ax.grid(alpha=0.2)
    return _save(run, fig, "accuracy_runtime")


def assemble_report(run):
    """Assemble observations from verified artifacts; optional timings are optional."""
    quality = quality_table(run)
    epochs = epoch_table(run)
    costs = timing_summary(run, epochs)
    benchmark = benchmark_table(run)

    def markdown(frame):
        if frame.empty:
            return "No saved rows."
        columns = list(frame.columns)

        def val(x):
            if x is None or pd.isna(x):
                return "not recorded"
            if isinstance(x, float):
                return f"{x:.5g}"
            return str(x).replace("|", "/").replace("\n", " ")

        return "\n".join(
            ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
            + [
                "| " + " | ".join(val(x) for x in row) + " |"
                for row in frame.itertuples(index=False, name=None)
            ]
        )

    highest = quality.accuracy.max()
    winners = quality.loc[quality.accuracy == highest, "variant"]
    winner_names = ", ".join(DISPLAY_NAMES[name] for name in winners)
    parameter_min = int(quality.parameter_count.min())
    parameter_max = int(quality.parameter_count.max())
    display_columns = [
        "variant",
        "seed",
        "parameter_count",
        "selected_epoch",
        "examples",
        "accuracy",
        "macro_f1",
        "loss",
        "majority_baseline_accuracy",
    ]
    displayed = quality[[key for key in display_columns if key in quality]].copy()
    for key in ("accuracy", "macro_f1", "majority_baseline_accuracy"):
        if key in displayed:
            displayed[key] = displayed[key].map(lambda value: f"{value:.2%}")
    text = f"""# EuroSAT: width versus depth

This report contains this run's saved results only.

Run: `{run.path.name}`; config SHA-256: `{run.identity["config_hash"]}`; backend: `{run.device}`.

## Observations

Highest observed test accuracy: **{winner_names}**, {highest:.2%}. Ties are retained. This is a descriptive result from one seed; it is not evidence of universal architectural superiority.

{markdown(displayed)}

Accuracy is the fraction of test images classified correctly. Macro F1 averages class-level F1 equally over all ten classes; larger is better. Cross-entropy loss is lower-is-better. The majority baseline always predicts the most frequent training class. Exact checkpoint and result identities remain in `../tables/quality.csv` and the evaluation manifests.

## Protocol and cost

EuroSAT RGB uses native 64×64 images and published TorchGeo longitude-based train/validation/test manifests. RGB statistics are fitted on training images only. Three residual-CNN shapes share the training recipe and contain **{parameter_min:,}–{parameter_max:,} parameters**, according to the saved selected models. Each fit starts from freshly seeded weights and uses {run.config.training.max_epochs} epochs, seed {run.config.training.seed}, AdamW, cosine learning-rate decay, FP32, batch {run.config.training.batch_size}. Validation accuracy selects each checkpoint; validation cross-entropy breaks ties, then the earlier epoch. All three checkpoints freeze before test access.

Epoch costs include train, validation, persistence and logging boundaries. Idle notebook time is excluded. Final receipt/table/TensorBoard timing-series publication is excluded explicitly. Missing receipts remain missing: `known_epoch_seconds` sums only available receipts, while `complete_fit_epoch_seconds` is reported only with every configured epoch present and timed. Memory readings are snapshots rather than peaks.

{markdown(costs)}

## Recovery provenance

"""
    for name in run.config.fits():
        path = run.fit_dir(name) / "attempts.json"
        text += f"- {DISPLAY_NAMES[name]}: "
        if path.exists():
            attempts = read_json(path)
            resumed = sum(attempt.get("resume_after_epoch", 0) > 0 for attempt in attempts)
            text += (
                f"{len(attempts)} recorded attempts; {resumed} started after a committed epoch. "
                f"[Saved attempt history](../models/{name}/seed_{run.config.training.seed}/attempts.json)."
            )
        else:
            text += "Attempt history is unavailable; do not infer uninterrupted execution."
        text += "\n"
    text += "\n## Figures\n\n"
    for name, title in [
        ("learning_curves", "Learning curves"),
        ("epoch_costs", "Epoch costs"),
        ("throughput", "Throughput"),
        ("memory", "Memory snapshots"),
        ("quality", "Held-out quality"),
        ("confusion_matrices", "Confusions"),
        ("accuracy_runtime", "Accuracy and runtime"),
    ]:
        if (run.path / "figures" / f"{name}.png").exists():
            text += f"### {title}\n\n![{title}](../figures/{name}.png)\n\n"
    text += "## Optional inference measurements\n\n"
    text += (
        markdown(benchmark)
        if not benchmark.empty
        else "Not executed. No inference-speed claim is made."
    )
    if not benchmark.empty:
        text += (
            "\n\nThese are synchronized, model-only forward timings on validation requests. "
            "Image decoding, normalization, host-to-device transfer and prediction formatting are excluded. "
            "One bounded block per model does not establish a paired optimization speedup; "
            "memory readings are separate snapshots."
        )
    review = run.path / "tables" / "failure_review.csv"
    if review.exists() and review.stat().st_size:
        text += "\n\n## Image-classification failure review\n\n"
        text += (
            "[Saved highest-confidence mistakes](../tables/failure_review.csv) were selected independently "
            "within each shape. This deliberate diagnostic sample is not a random estimate of error types. "
            "The full saved predictions retain all test mistakes. Do not tune using these test errors."
        )
    text += """

## Interpretation and limits

Equal parameter counts do not equal equal compute, receptive field, optimization difficulty or regularization. Any proposed explanation for a difference remains a hypothesis. One seed and one training recipe cannot establish statistical superiority; there are no seed error bars. The published spatial split reduces random-image mixing but is not a buffered or universal geographic-independence guarantee. Results describe ten land-cover categories from RGB satellite patches, not arbitrary vision tasks, multispectral reasoning or deployment safety. Runtime and memory apply to this backend and batch configuration. Do not retune after inspecting test failures.

Dataset provenance: [EuroSAT creators](https://github.com/phelber/EuroSAT), [RGB archive](https://zenodo.org/records/7711810), [TorchGeo spatial split](https://docs.torchgeo.org/en/stable/api/datasets/eurosat.html). Source, configuration, split identities and exact versions are saved with this run.
"""
    destination = run.path / "report" / "technical_report.md"
    with run.writer_lock():
        atomic_text(destination, text)
    return destination
