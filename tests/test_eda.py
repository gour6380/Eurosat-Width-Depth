"""EDA never requests held-out pixels or uses stochastic example selection."""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eda import (
    class_distribution,
    dataset_overview,
    plot_training_rgb,
    show_training_examples,
    training_example_indices,
)


class TrainingOnlyData:
    class_names = ("Forest", "River")
    mean = (0.1, 0.2, 0.3)
    std = (0.05, 0.06, 0.07)
    identity = "fixture"
    manifest = {
        "dataset": "fixture",
        "source": "fixture",
        "split_method": "fixture spatial",
        "splits": {
            split: {"count": 4, "class_counts": {"Forest": 2, "River": 2}}
            for split in ("train", "val", "test")
        },
    }

    def sample_ids(self, split):
        assert split == "train"
        return ["Forest_9.jpg", "River_2.jpg", "Forest_1.jpg", "River_7.jpg"]

    def dataset(self, split, **kwargs):
        assert split == "train" and not kwargs
        return [
            (torch.full((3, 64, 64), i * 20, dtype=torch.uint8), label)
            for i, label in enumerate((0, 1, 0, 1))
        ]


def test_selection_deterministic_and_rng_untouched():
    data = TrainingOnlyData()
    state = np.random.get_state()
    assert training_example_indices(data) == [2, 0, 1, 3]
    assert training_example_indices(data) == [2, 0, 1, 3]
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_all_eda_helpers_use_training_pixels_only(tmp_path):
    run = SimpleNamespace(path=tmp_path)
    data = TrainingOnlyData()
    overview = dataset_overview(run, data)
    assert "test images" in overview["field"].tolist()
    table, _ = class_distribution(run, data)
    assert table["train"].sum() == 4
    show_training_examples(run, data)
    moments, _ = plot_training_rgb(run, data)
    assert moments["mean_rgb_div_255"].tolist() == list(data.mean)
    assert len(list((tmp_path / "figures" / "eda").glob("*.png"))) == 3
    plt.close("all")
