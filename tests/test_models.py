"""Model acceptance tests; written for the owner's later runtime validation."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from src.config import ExperimentConfig, ModelConfig
from src.models import analytical_parameter_count, architecture_table, build_model


@pytest.mark.parametrize(
    ("name", "blocks", "width", "count"),
    [
        ("shallow_wide", 1, 48, 691_834),
        ("middle", 2, 32, 696_618),
        ("deep_narrow", 4, 22, 697_344),
    ],
)
def test_exact_counts_and_output_shape(name, blocks, width, count):
    shape = ModelConfig(name, blocks, width)
    model = build_model(shape).eval()
    assert analytical_parameter_count(shape) == count
    assert sum(p.numel() for p in model.parameters()) == count
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 10)


def test_generic_class_count_matches_formula():
    shape = ModelConfig("middle", 2, 8)
    model = build_model(shape, classes=3)
    assert sum(p.numel() for p in model.parameters()) == analytical_parameter_count(shape, 3)


def test_stage_shapes_and_projection_gradients():
    model = build_model(ModelConfig("middle", 2, 8))
    inputs = torch.randn(2, 3, 64, 64)
    hidden = model.stem(inputs)
    for stage, expected in zip(
        model.stages, [(8, 64, 64), (16, 32, 32), (32, 16, 16)], strict=True
    ):
        hidden = stage(hidden)
        assert tuple(hidden.shape[1:]) == expected
    loss = nn.functional.cross_entropy(model(inputs), torch.tensor([1, 2]))
    loss.backward()
    for stage in (model.stages[1], model.stages[2]):
        gradient = stage[0].shortcut[0].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_blocks_have_independent_weights():
    torch.manual_seed(42)
    model = build_model(ModelConfig("middle", 2, 8))
    first, second = model.stages[0][0].conv1.weight, model.stages[0][1].conv1.weight
    assert first.shape == second.shape
    assert first.data_ptr() != second.data_ptr()
    assert not torch.equal(first, second)
    assert not any(isinstance(layer, nn.Dropout) for layer in model.modules())


def test_initialization_is_reproducible_for_same_shape_and_seed():
    shape = ModelConfig("middle", 2, 8)
    torch.manual_seed(42)
    first = build_model(shape)
    torch.manual_seed(42)
    second = build_model(shape)
    for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        assert torch.equal(left, right)


def test_architecture_comparison_preserves_rng_and_matches_capacity():
    config = ExperimentConfig()
    torch.manual_seed(43)
    before = torch.get_rng_state().clone()
    table = architecture_table(config)
    assert torch.equal(before, torch.get_rng_state())
    assert list(table["parameters"]) == [691_834, 696_618, 697_344]
    assert table["comparison_parameter_spread"].iloc[0] < 0.01
    assert (
        table["stem_parameters"]
        + table["processing_block_parameters"]
        + table["classifier_parameters"]
    ).equals(table["parameters"])


def test_architecture_comparison_rejects_unmatched_settings():
    config = ExperimentConfig()
    changed = (replace(config.models[0], base_width=16), *config.models[1:])
    with pytest.raises(ValueError, match="Parameter spread"):
        architecture_table(replace(config, models=changed))


@pytest.mark.parametrize("blocks,width,classes", [(0, 8, 10), (1, 0, 10), (1, 8, 0), (True, 8, 10)])
def test_invalid_architecture_rejected(blocks, width, classes):
    with pytest.raises(ValueError, match="positive integers"):
        analytical_parameter_count(ModelConfig("middle", blocks, width), classes)


def test_input_channel_validation():
    model = build_model(ModelConfig("middle", 1, 8))
    with pytest.raises(ValueError, match="RGB"):
        model(torch.zeros(2, 1, 64, 64))
