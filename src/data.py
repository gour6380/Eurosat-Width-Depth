"""Verified EuroSAT RGB cache and published longitude-based spatial partitions.

Downloads and pixel decoding occur only when ``prepare_data`` is called. Images
are never resized. Normalization statistics use the training partition only.
"""

import hashlib
import io
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import DataConfig
from .io import atomic_json, canonical_hash, filesystem_lock, read_json, sha256_file

CACHE_VERSION = "eurosat-rgb-spatial-v1"
SPLIT_SHA256 = {
    "train": "2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f",
    "val": "6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b",
    "test": "de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452",
}
CLASS_NAMES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(url: str, path: Path, checksum: str, algorithm: str) -> None:
    if path.exists():
        if _digest(path, algorithm) != checksum:
            raise RuntimeError(
                f"Corrupt existing download: {path}; remove it deliberately before retrying"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            request = Request(url, headers={"User-Agent": "EuroSAT-width-depth-research/1.0"})
            _stream_download(request, output, path.name)
            output.flush()
            os.fsync(output.fileno())
        if _digest(Path(temporary), algorithm) != checksum:
            raise RuntimeError(f"Published {algorithm} mismatch for {url}")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _stream_download(request, output, name: str) -> None:
    with urlopen(request, timeout=60) as response:
        total = response.headers.get("Content-Length")
        with tqdm(
            total=int(total) if total else None, unit="B", unit_scale=True, desc=name
        ) as progress:
            while block := response.read(1024 * 1024):
                output.write(block)
                progress.update(len(block))


def _parse_split(text: str) -> list[str]:
    result = []
    for raw in text.splitlines():
        value = raw.strip()
        if not value:
            continue
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError(f"Unsafe split entry: {value}")
        name = path.name
        if path.suffix.lower() != ".jpg" or name.split("_", 1)[0] not in CLASS_NAMES:
            raise ValueError(f"Unexpected split entry: {value}")
        result.append(name)
    if not result or len(result) != len(set(result)):
        raise ValueError("Split must be nonempty and contain no duplicate image IDs")
    return result


def _validate_partitions(
    partitions: dict[str, list[str]], image_ids: set[str], expected_images: int
) -> None:
    union = set()
    for split in SPLIT_SHA256:
        ids = partitions[split]
        if len(ids) != len(set(ids)) or union.intersection(ids):
            raise ValueError(f"Duplicate or cross-partition image in {split}")
        observed = {name.split("_", 1)[0] for name in ids}
        if observed != set(CLASS_NAMES):
            raise ValueError(f"Every class must occur in the {split} partition")
        union.update(ids)
    if union != image_ids or len(union) != expected_images:
        raise ValueError("Published partitions must cover the complete RGB archive exactly once")


def _zip_images(archive: ZipFile) -> dict[str, str]:
    result = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"Unsafe archive member: {info.filename}")
        if info.is_dir() or path.suffix.lower() != ".jpg":
            continue
        # Read JPEG bytes directly; never extract the archive to the filesystem.
        name = path.name
        label = name.split("_", 1)[0]
        if label not in CLASS_NAMES or path.parent.name != label:
            raise ValueError(f"Unexpected class/path in archive: {info.filename}")
        if name in result or info.file_size > 1024 * 1024:
            raise ValueError(f"Duplicate or oversized JPEG: {name}")
        result[name] = info.filename
    return result


def _build_cache(destination: Path, raw: Path, config, partitions, archive_path: Path) -> dict:
    files = {}
    split_info = {}
    sums = np.zeros(3, dtype=np.float64)
    squared = np.zeros(3, dtype=np.float64)
    pixels = 0
    with ZipFile(archive_path) as archive:
        members = _zip_images(archive)
        _validate_partitions(partitions, set(members), config.expected_images)
        for split, ids in partitions.items():
            images_path = destination / f"{split}_images.npy"
            images = np.lib.format.open_memmap(
                images_path,
                mode="w+",
                dtype=np.uint8,
                shape=(len(ids), 3, config.image_size, config.image_size),
            )
            labels = np.empty(len(ids), dtype=np.int64)
            for index, image_id in enumerate(tqdm(ids, desc=f"Decode {split}", unit="image")):
                with Image.open(io.BytesIO(archive.read(members[image_id]))) as image:
                    if image.mode != "RGB" or image.size != (config.image_size, config.image_size):
                        raise ValueError(f"Unexpected image mode or dimensions: {image_id}")
                    values = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
                images[index] = values
                labels[index] = CLASS_NAMES.index(image_id.split("_", 1)[0])
                if split == "train":
                    unit = values.astype(np.float64) / 255.0
                    sums += unit.sum(axis=(1, 2))
                    squared += np.square(unit).sum(axis=(1, 2))
                    pixels += config.image_size**2
            images.flush()
            del images
            np.save(destination / f"{split}_labels.npy", labels, allow_pickle=False)
            atomic_json(destination / f"{split}_ids.json", ids)
            split_info[split] = {
                "count": len(ids),
                "shape": [len(ids), 3, config.image_size, config.image_size],
                "class_counts": dict(
                    zip(CLASS_NAMES, np.bincount(labels, minlength=10).tolist(), strict=True)
                ),
                "ids_sha256": canonical_hash(ids),
            }
            for suffix in ("images.npy", "labels.npy", "ids.json"):
                filename = f"{split}_{suffix}"
                files[filename] = sha256_file(destination / filename)
    mean = sums / pixels
    variance = np.maximum(squared / pixels - mean**2, 0)
    std = np.sqrt(variance)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid training-only RGB normalization statistics")
    manifest = {
        "cache_version": CACHE_VERSION,
        "data_config": asdict(config),
        "dataset": "EuroSAT RGB",
        "source": config.archive_url,
        "archive_sha256": sha256_file(archive_path),
        "archive_md5": config.archive_md5,
        "split_source": config.split_base_url,
        "split_sha256": {
            split: sha256_file(raw / f"eurosat-spatial-{split}.txt") for split in SPLIT_SHA256
        },
        "split_method": "Published TorchGeo longitude-based EuroSATSpatial partitions",
        "class_names": list(CLASS_NAMES),
        "splits": split_info,
        "normalization": {
            "fitted_partition": "train",
            "units": "RGB / 255",
            "mean": mean.tolist(),
            "std": std.tolist(),
            "pixels_per_channel": pixels,
        },
        "files": files,
        "limitations": "Longitude partitioning is not a guarantee of complete spatial independence or a geographic buffer.",
    }
    manifest["identity"] = canonical_hash(manifest)
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def _verify_cache(root: Path, config) -> dict:
    manifest = read_json(root / "manifest.json")
    contents = {k: v for k, v in manifest.items() if k != "identity"}
    if manifest["identity"] != canonical_hash(contents):
        raise RuntimeError("Dataset manifest identity does not match its contents")
    if manifest["cache_version"] != CACHE_VERSION or manifest["data_config"] != asdict(config):
        raise RuntimeError("Dataset cache belongs to different processing settings")
    if manifest["split_sha256"] != SPLIT_SHA256:
        raise RuntimeError("Dataset cache does not use the pinned spatial splits")
    for filename, digest in manifest["files"].items():
        if Path(filename).name != filename or sha256_file(root / filename) != digest:
            raise RuntimeError(f"Dataset cache checksum mismatch: {filename}")
    for split in SPLIT_SHA256:
        images = np.load(root / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(root / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)
        ids = read_json(root / f"{split}_ids.json")
        if (
            images.dtype != np.uint8
            or list(images.shape) != manifest["splits"][split]["shape"]
            or labels.dtype != np.int64
            or labels.shape != (len(ids),)
            or len(ids) != len(images)
            or canonical_hash(ids) != manifest["splits"][split]["ids_sha256"]
        ):
            raise RuntimeError(f"Dataset array structure mismatch: {split}")
    return manifest


class ImageDataset(Dataset):
    """Memory-mapped RGB arrays; no augmentation or normalization on the CPU."""

    def __init__(self, root: Path, split: str):
        self.images = np.load(root / f"{split}_images.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(root / f"{split}_labels.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # The small copy gives PyTorch a writable tensor, preserving immutable cache bytes.
        return torch.from_numpy(self.images[index].copy()), int(self.labels[index])


@dataclass
class PreparedData:
    root: Path
    identity: str
    manifest: dict

    def verify(self) -> None:
        verified = _verify_cache(self.root, DataConfig(**self.manifest["data_config"]))
        if verified["identity"] != self.identity:
            raise RuntimeError("Prepared dataset identity changed after preparation")

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["class_names"])

    @property
    def mean(self) -> tuple[float, ...]:
        return tuple(self.manifest["normalization"]["mean"])

    @property
    def std(self) -> tuple[float, ...]:
        return tuple(self.manifest["normalization"]["std"])

    def dataset(self, split: str, *, allow_test: bool = False) -> ImageDataset:
        if split not in SPLIT_SHA256:
            raise ValueError(f"Unknown partition: {split}")
        if split == "test" and not allow_test:
            raise RuntimeError(
                "Test pixels are reserved for evaluation after all three fits are frozen"
            )
        return ImageDataset(self.root, split)

    def sample_ids(self, split: str) -> list[str]:
        if split not in SPLIT_SHA256:
            raise ValueError(f"Unknown partition: {split}")
        return read_json(self.root / f"{split}_ids.json")


def prepare_data(run) -> PreparedData:
    """Download/validate once, then reuse a checksummed, train-normalized cache."""
    run.verify()
    config = run.config.data
    cache_key = canonical_hash({"version": CACHE_VERSION, "config": asdict(config)})
    data_root = run.root / "data"
    raw = data_root / "raw"
    destination = data_root / "processed" / cache_key
    with filesystem_lock(data_root / ".prepare.lock"):
        _download_verified(config.archive_url, raw / "EuroSAT_RGB.zip", config.archive_md5, "md5")
        for split, digest in SPLIT_SHA256.items():
            filename = f"eurosat-spatial-{split}.txt"
            _download_verified(config.split_base_url + filename, raw / filename, digest, "sha256")
        if destination.exists():
            manifest = _verify_cache(destination, config)
            if manifest["archive_sha256"] != sha256_file(raw / "EuroSAT_RGB.zip"):
                raise RuntimeError("Raw archive identity differs from the processed cache")
        else:
            partitions = {
                split: _parse_split((raw / f"eurosat-spatial-{split}.txt").read_text())
                for split in SPLIT_SHA256
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=destination.parent))
            try:
                manifest = _build_cache(temporary, raw, config, partitions, raw / "EuroSAT_RGB.zip")
                _verify_cache(temporary, config)
                os.rename(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    run.bind_data(manifest["identity"], destination / "manifest.json")
    return PreparedData(destination, manifest["identity"], manifest)
