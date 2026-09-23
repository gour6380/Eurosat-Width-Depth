"""Saved training-only EDA; these helpers never train, benchmark, or evaluate."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import atomic_csv


def _locations(run):
    figures = run.path / "figures" / "eda"
    tables = run.path / "data_manifest" / "eda"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def _save(fig, path: Path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    return fig


def dataset_overview(run, data) -> pd.DataFrame:
    """Metadata for all partitions; no validation/test pixel inspection."""
    _, tables = _locations(run)
    rows = [
        {"field": "Dataset", "value": data.manifest["dataset"]},
        {"field": "Source", "value": data.manifest["source"]},
        {"field": "Spatial split", "value": data.manifest["split_method"]},
        {"field": "Image dimensions", "value": "3 × 64 × 64; native RGB, no resize"},
        {"field": "Image values", "value": "Cached uint8 [0,255]; device normalization after /255"},
        {"field": "Classes", "value": len(data.class_names)},
        {"field": "Normalization fit", "value": "Training pixels only"},
        {"field": "Dataset identity", "value": data.identity},
    ]
    rows.extend(
        {"field": f"{split} images", "value": meta["count"]}
        for split, meta in data.manifest["splits"].items()
    )
    atomic_csv(tables / "dataset_overview.csv", rows)
    return pd.DataFrame(rows)


def class_distribution(run, data):
    """Class counts are split metadata; plot training counts only."""
    figures, tables = _locations(run)
    rows = [
        {
            "class": label,
            **{
                split: meta["class_counts"][label]
                for split, meta in data.manifest["splits"].items()
            },
        }
        for label in data.class_names
    ]
    atomic_csv(tables / "class_counts.csv", rows)
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    bars = ax.barh(frame["class"], frame["train"], color="#356BA2")
    ax.bar_label(bars, padding=5, fontsize=10)
    ax.set_xlim(0, frame["train"].max() * 1.18)
    ax.set_xlabel("Training images (count)")
    ax.set_title("Training coverage across the ten EuroSAT classes", pad=15)
    ax.invert_yaxis()
    return frame, _save(fig, figures / "training_class_distribution.png")


def training_example_indices(data, examples_per_class: int = 2) -> list[int]:
    """Deterministic filename order, independent of model results and RNG state."""
    if examples_per_class < 1 or examples_per_class > 5:
        raise ValueError("Choose between one and five examples per class")
    ids = data.sample_ids("train")
    result = []
    for label in data.class_names:
        candidates = sorted(
            (name, index) for index, name in enumerate(ids) if name.split("_", 1)[0] == label
        )
        if len(candidates) < examples_per_class:
            raise ValueError(f"Insufficient training examples for {label}")
        # Spread illustrations across the sorted identifiers; no target/quality filtering.
        positions = np.linspace(0, len(candidates) - 1, examples_per_class, dtype=int)
        result.extend(candidates[position][1] for position in positions)
    return result


def show_training_examples(run, data, examples_per_class: int = 2):
    figures, tables = _locations(run)
    indices = training_example_indices(data, examples_per_class)
    dataset = data.dataset("train")
    ids = data.sample_ids("train")
    fig, axes = plt.subplots(
        len(data.class_names),
        examples_per_class,
        figsize=(examples_per_class * 4.2, 20),
        squeeze=False,
        layout="constrained",
    )
    rows = []
    for ax, index in zip(axes.flat, indices, strict=True):
        image, label = dataset[index]
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(ids[index], fontsize=11)
        ax.axis("off")
        rows.append(
            {"image_id": ids[index], "class": data.class_names[label], "partition": "train"}
        )
    fig.suptitle("Deterministic training examples — native 64 × 64 RGB", fontsize=16)
    atomic_csv(tables / "training_examples.csv", rows)
    return _save(fig, figures / "training_examples.png")


def plot_training_rgb(run, data):
    """All-pixel train-only moments and a deterministic 512-image histogram."""
    figures, tables = _locations(run)
    dataset = data.dataset("train")
    indices = np.linspace(0, len(dataset) - 1, min(512, len(dataset)), dtype=int)
    counts = np.zeros((3, 256), dtype=np.int64)
    for index in indices:
        image, _ = dataset[int(index)]
        for channel in range(3):
            counts[channel] += np.bincount(image[channel].numpy().ravel(), minlength=256)
    rows = [
        {
            "channel": name,
            "mean_rgb_div_255": data.mean[i],
            "std_rgb_div_255": data.std[i],
            "fitted_partition": "train",
        }
        for i, name in enumerate(("Red", "Green", "Blue"))
    ]
    atomic_csv(tables / "training_rgb_statistics.csv", rows)
    atomic_csv(
        tables / "training_rgb_histogram.csv",
        [
            {
                "uint8_value": value,
                **{name: int(counts[i, value]) for i, name in enumerate(("red", "green", "blue"))},
            }
            for value in range(256)
        ],
    )
    ids = data.sample_ids("train")
    atomic_csv(
        tables / "training_rgb_histogram_images.csv", [{"image_id": ids[int(i)]} for i in indices]
    )
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for channel, (name, color) in enumerate(
        (("Red", "#C33C44"), ("Green", "#26834A"), ("Blue", "#356BA2"))
    ):
        ax.plot(np.arange(256), counts[channel] / counts[channel].sum(), label=name, color=color)
    ax.set_xlabel("Stored RGB channel value")
    ax.set_ylabel("Fraction of sampled channel pixels")
    ax.set_title(f"Training RGB distribution — {len(indices)} deterministic images", pad=15)
    ax.legend(title="Channel", frameon=False)
    ax.grid(alpha=0.15)
    return pd.DataFrame(rows), _save(fig, figures / "training_rgb_distribution.png")
