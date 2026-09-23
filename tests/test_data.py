"""Small synthetic fixtures; tests never download the real dataset."""

import io
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from src.config import DataConfig
from src.data import (
    CLASS_NAMES,
    PreparedData,
    _build_cache,
    _download_verified,
    _parse_split,
    _validate_partitions,
    _verify_cache,
    _zip_images,
)


def partitions():
    return {
        split: [f"{label}_{index}.jpg" for label in CLASS_NAMES]
        for index, split in enumerate(("train", "val", "test"), 1)
    }


def test_split_paths_and_duplicate_rejection():
    assert _parse_split("AnnualCrop/AnnualCrop_1.jpg\nForest_2.jpg\n") == [
        "AnnualCrop_1.jpg",
        "Forest_2.jpg",
    ]
    for bad in ("../Forest_1.jpg", "/Forest_1.jpg", "Forest_1.jpg\nForest_1.jpg", "Unknown_1.jpg"):
        with pytest.raises(ValueError):
            _parse_split(bad)


def test_partitions_disjoint_complete_and_all_classes():
    parts = partitions()
    ids = set(sum(parts.values(), []))
    _validate_partitions(parts, ids, 30)
    parts["test"][0] = parts["train"][0]
    with pytest.raises(ValueError, match="cross-partition"):
        _validate_partitions(parts, ids, 30)
    parts = partitions()
    with pytest.raises(ValueError, match="complete RGB"):
        _validate_partitions(parts, ids | {"Forest_999.jpg"}, 31)


def test_zip_duplicate_and_unsafe_members(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../Forest/Forest_1.jpg", b"bad")
    with ZipFile(archive_path) as archive, pytest.raises(ValueError, match="Unsafe"):
        _zip_images(archive)


def test_download_existing_bad_checksum_stops_without_network(tmp_path, monkeypatch):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"bad")
    monkeypatch.setattr("src.data.urlopen", lambda *a, **kw: pytest.fail("Must not access network"))
    with pytest.raises(RuntimeError, match="Corrupt existing"):
        _download_verified("https://example.invalid", path, "0" * 32, "md5")


def _tiny_cache(tmp_path):
    destination = tmp_path / "cache"
    destination.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    parts = partitions()
    path = raw / "EuroSAT_RGB.zip"
    with ZipFile(path, "w") as archive:
        for split, ids in parts.items():
            (raw / f"eurosat-spatial-{split}.txt").write_text("\n".join(ids))
            for class_index, name in enumerate(ids):
                # Nonzero variation in training; validation/test deliberately much brighter.
                level = 10 + class_index * 2 if split == "train" else 200
                buffer = io.BytesIO()
                Image.new("RGB", (64, 64), color=(level, level, level)).save(buffer, format="JPEG")
                archive.writestr(f"EuroSAT_RGB/{name.split('_')[0]}/{name}", buffer.getvalue())
    config = replace(DataConfig(), expected_images=30)
    manifest = _build_cache(destination, raw, config, parts, path)
    return destination, manifest, config


def test_native_arrays_and_training_only_statistics(tmp_path):
    root, manifest, _ = _tiny_cache(tmp_path)
    data = PreparedData(root, manifest["identity"], manifest)
    image, label = data.dataset("train")[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert image.numpy().dtype == np.uint8 and label == 0
    assert data.mean == pytest.approx((19 / 255,) * 3, abs=1e-6)
    assert all(0 < x < 0.1 for x in data.std)
    with pytest.raises(RuntimeError, match="frozen"):
        data.dataset("test")
    assert len(data.dataset("test", allow_test=True)) == 10
    assert data.sample_ids("val")[0] == "AnnualCrop_2.jpg"


def test_cache_identity_rejects_edited_manifest(tmp_path):
    root, manifest, config = _tiny_cache(tmp_path)
    manifest["normalization"]["mean"][0] = 0.8
    from src.io import atomic_json

    atomic_json(root / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="identity"):
        _verify_cache(root, config)


def test_fixture_pinned_identity_and_byte_corruption(tmp_path, monkeypatch):
    root, manifest, config = _tiny_cache(tmp_path)
    monkeypatch.setattr("src.data.SPLIT_SHA256", manifest["split_sha256"])
    assert _verify_cache(root, config)["identity"] == manifest["identity"]
    path = root / "val_labels.npy"
    with Path(path).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        _verify_cache(root, config)
