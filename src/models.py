"""Compact residual CNNs differing only in stage width and repeated block count.

Every convolution is bias-free and followed by BatchNorm. A block has two 3x3
convolutions; a 1x1 projection with BatchNorm handles a stage's stride/width
change. Parameter counts include BatchNorm's learned scale and offset, but not
its running-statistic buffers. The classifier is biased. With ``n`` blocks per
stage, base width ``b`` and ``c`` classes, the exact count is

    (378*n - 80)*b**2 + (28*n + 41 + 4*c)*b + c.

Matching this count controls capacity approximately, not compute or receptive
field. No project model is constructed merely by importing this module.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import torch
from torch import nn

from .config import ExperimentConfig, ModelConfig

DISPLAY_NAMES = {
    "shallow_wide": "Shallow / wide",
    "middle": "Middle",
    "deep_narrow": "Deep / narrow",
}


def _validate_dimensions(config: ModelConfig, classes: int) -> None:
    values = (config.blocks_per_stage, config.base_width, classes)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Block counts, widths and class count must be positive integers")


def analytical_parameter_count(config: ModelConfig, classes: int = 10) -> int:
    """Exact count for this block definition, without constructing tensors."""
    _validate_dimensions(config, classes)
    n, b = config.blocks_per_stage, config.base_width
    return (378 * n - 80) * b**2 + (28 * n + 41 + 4 * classes) * b + classes


class ResidualBlock(nn.Module):
    """Post-activation basic residual block; projection only when necessary."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.bn1(self.conv1(inputs)))
        hidden = self.bn2(self.conv2(hidden))
        return self.relu(hidden + self.shortcut(inputs))


class ResidualClassifier(nn.Module):
    """Three-stage CNN for native-resolution RGB satellite images.

    At 64x64 input resolution, stage feature maps are 64x64, 32x32 and 16x16.
    The main path contains 1 + 6*n convolutions and one linear classifier.
    Two projection convolutions are additional shortcut operations. Layers are
    initialized separately; training code owns the random seed and device.
    """

    def __init__(self, config: ModelConfig, classes: int = 10):
        super().__init__()
        _validate_dimensions(config, classes)
        self.architecture_config = asdict(config)
        self.classes = classes
        n, b = config.blocks_per_stage, config.base_width
        self.stem = nn.Sequential(
            nn.Conv2d(3, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=False),
        )
        stages = []
        incoming = b
        for stage_index, width in enumerate((b, 2 * b, 4 * b)):
            blocks = []
            for block_index in range(n):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(ResidualBlock(incoming, width, stride))
                incoming = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4 * b, classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("Expected a batch of RGB images with shape [N, 3, H, W]")
        features = self.stages(self.stem(inputs))
        return self.classifier(self.pool(features).flatten(1))


def build_model(config: ModelConfig, classes: int = 10) -> ResidualClassifier:
    """Construct fresh FP32 CPU weights; the caller explicitly chooses device."""
    with torch.device("cpu"):
        model = ResidualClassifier(config, classes).float()
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = analytical_parameter_count(config, classes)
    if actual != expected:
        raise RuntimeError(f"Architecture parameter-count mismatch: {actual} != {expected}")
    return model


def architecture_table(config: ExperimentConfig) -> pd.DataFrame:
    """Inspect models one at a time on CPU and reject an unmatched comparison.

    This function intentionally constructs models when explicitly called. It
    restores CPU RNG state so notebook inspection cannot consume the training
    stream. Training still seeds every fit independently.
    """
    config.validate()
    rows = []
    with torch.random.fork_rng(devices=[]):
        for shape in config.models:
            model = build_model(shape, config.data.classes)
            count = sum(parameter.numel() for parameter in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            stem = sum(p.numel() for p in model.stem.parameters())
            blocks = sum(p.numel() for p in model.stages.parameters())
            head = sum(p.numel() for p in model.classifier.parameters())
            rows.append(
                {
                    "variant": shape.name,
                    "display_name": DISPLAY_NAMES[shape.name],
                    "blocks_per_stage": shape.blocks_per_stage,
                    "residual_blocks": 3 * shape.blocks_per_stage,
                    "main_path_convolutions": 1 + 6 * shape.blocks_per_stage,
                    "shortcut_convolutions": 2,
                    "base_width": shape.base_width,
                    "stage_widths": "/".join(str(shape.base_width * k) for k in (1, 2, 4)),
                    "parameters": count,
                    "trainable_parameters": trainable,
                    "stem_parameters": stem,
                    "processing_block_parameters": blocks,
                    "classifier_parameters": head,
                    "parameter_bytes_fp32": count * 4,
                }
            )
            del model
    counts = [row["parameters"] for row in rows]
    spread = (max(counts) - min(counts)) / min(counts)
    if spread > config.max_parameter_spread:
        raise ValueError(
            f"Parameter spread {spread:.2%} exceeds the configured "
            f"{config.max_parameter_spread:.2%} limit"
        )
    table = pd.DataFrame(rows)
    table["comparison_parameter_spread"] = spread
    return table
