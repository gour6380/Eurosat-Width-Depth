"""One notebook-owned, immutable experiment configuration."""

import math
from dataclasses import asdict, dataclass, field

PROTOCOL = "eurosat-width-depth-v3"


@dataclass(frozen=True)
class DataConfig:
    archive_url: str = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
    archive_md5: str = "f46e308c4d50d4bf32fedad2d3d62f3b"
    split_base_url: str = (
        "https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/"
    )
    image_size: int = 64
    expected_images: int = 27_000
    classes: int = 10


@dataclass(frozen=True)
class ModelConfig:
    name: str
    blocks_per_stage: int
    base_width: int


def default_models() -> tuple[ModelConfig, ...]:
    # Final widths are set from the analytical count of the actual block definition.
    return (
        ModelConfig("shallow_wide", 1, 48),
        ModelConfig("middle", 2, 32),
        ModelConfig("deep_narrow", 4, 22),
    )


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    max_epochs: int = 20
    batch_size: int = 128
    validation_batch_size: int = 256
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    num_workers: int = 0
    horizontal_flip: bool = True
    vertical_flip: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_sizes: tuple[int, ...] = (1, 64)
    warmups: int = 5
    repetitions: int = 30


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    models: tuple[ModelConfig, ...] = field(default_factory=default_models)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    max_parameter_spread: float = 0.05
    protocol: str = PROTOCOL

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        value = dict(data)
        if value.get("protocol") != PROTOCOL:
            raise ValueError(
                "Saved protocol differs; create a fresh run or use its matching source"
            )
        value["data"] = DataConfig(**value["data"])
        value["training"] = TrainingConfig(**value["training"])
        value["models"] = tuple(ModelConfig(**x) for x in value["models"])
        bench = dict(value["benchmark"])
        bench["batch_sizes"] = tuple(bench["batch_sizes"])
        value["benchmark"] = BenchmarkConfig(**bench)
        result = cls(**value)
        result.validate()
        return result

    def model(self, name: str) -> ModelConfig:
        return next(x for x in self.models if x.name == name)

    def fits(self) -> tuple[str, ...]:
        return tuple(x.name for x in self.models)

    def validate(self) -> None:
        if self.protocol != PROTOCOL:
            raise ValueError("Use the matching source for this saved protocol")
        if self.fits() != ("shallow_wide", "middle", "deep_narrow"):
            raise ValueError("The comparison requires the three registered shapes")
        if any(
            type(v) is not int or v < 1
            for m in self.models
            for v in (m.blocks_per_stage, m.base_width)
        ):
            raise ValueError("Model dimensions must be positive")
        if (self.data.image_size, self.data.expected_images, self.data.classes) != (64, 27_000, 10):
            raise ValueError(
                "This protocol uses the complete native-resolution EuroSAT RGB archive"
            )
        t = self.training
        if any(
            type(v) is not int or v < 1
            for v in (t.max_epochs, t.batch_size, t.validation_batch_size)
        ):
            raise ValueError("Epoch and batch counts must be positive")
        if type(t.seed) is not int or not 0 <= t.seed < 2**32 or t.num_workers != 0:
            raise ValueError("Use one nonnegative integer seed and workers=0")
        if any(type(v) is not bool for v in (t.horizontal_flip, t.vertical_flip)):
            raise ValueError("Augmentation switches must be Boolean")
        numbers = (
            t.learning_rate,
            t.min_learning_rate,
            t.weight_decay,
            t.gradient_clip,
            self.max_parameter_spread,
        )
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError("Configuration contains a nonfinite value")
        if (
            not 0 < t.min_learning_rate <= t.learning_rate
            or t.weight_decay < 0
            or t.gradient_clip <= 0
        ):
            raise ValueError("Invalid fixed AdamW recipe")
        if not 0 < self.max_parameter_spread <= 0.05:
            raise ValueError("Invalid parameter-matching tolerance")
        b = self.benchmark
        if b.batch_sizes != (1, 64) or any(
            type(v) is not int or v < 1 for v in (b.warmups, b.repetitions)
        ):
            raise ValueError("Use the bounded B1/B64 inference benchmark")
