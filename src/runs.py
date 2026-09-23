"""Explicit numbered runs, frozen experiment identity and cumulative fit manifests."""

import importlib.metadata
import platform
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import ExperimentConfig
from .io import atomic_json, canonical_hash, filesystem_lock, read_json, sha256_file
from .runtime import select_device


def source_manifest(root):
    paths = sorted((root / "src").rglob("*.py"))
    paths += [root / name for name in ("pyproject.toml", "uv.lock") if (root / name).exists()]
    return {str(p.relative_to(root)): sha256_file(p) for p in paths}


def environment_manifest():
    # Installed versions and interpreter/platform are identity; notebook UI state is not.
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(
            sorted(
                (d.metadata["Name"].lower(), d.version)
                for d in importlib.metadata.distributions()
                if d.metadata.get("Name")
            )
        ),
    }


@dataclass
class Run:
    root: Path
    path: Path
    config: ExperimentConfig
    device: object
    identity: dict

    def writer_lock(self):
        return filesystem_lock(self.path / ".writer.lock")

    @property
    def data_identity(self):
        p = self.path / "data_manifest" / "binding.json"
        return read_json(p)["identity"] if p.exists() else None

    def fit_dir(self, name):
        if name not in self.config.fits():
            raise ValueError("Unregistered fit")
        return self.path / "models" / name / f"seed_{self.config.training.seed}"

    def verify(self, data=None):
        if read_json(self.path / "identity.json") != self.identity:
            raise RuntimeError("Run identity changed or is corrupt")
        if canonical_hash(read_json(self.path / "config.json")) != self.identity["config_hash"]:
            raise RuntimeError("Saved configuration is corrupt")
        if canonical_hash(self.config.to_dict()) != self.identity["config_hash"]:
            raise RuntimeError("Notebook configuration differs; create a fresh run")
        current = source_manifest(self.root)
        if (
            canonical_hash(read_json(self.path / "source_manifest.json"))
            != self.identity["source_hash"]
        ):
            raise RuntimeError("Saved source manifest is corrupt")
        if canonical_hash(current) != self.identity["source_hash"]:
            raise RuntimeError("Source/dependency lock changed; create a fresh run")
        for rel, checksum in current.items():
            saved = self.path / "source_snapshot" / rel
            if not saved.is_file() or sha256_file(saved) != checksum:
                raise RuntimeError(f"Source snapshot is corrupt: {rel}")
        env = environment_manifest()
        if canonical_hash(env) != self.identity["environment_hash"]:
            raise RuntimeError("Environment changed; restore it or create a fresh run")
        if (
            canonical_hash(read_json(self.path / "environment.json"))
            != self.identity["environment_hash"]
        ):
            raise RuntimeError("Saved environment identity is corrupt")
        if str(self.device) != self.identity["device"] or str(select_device()) != str(self.device):
            raise RuntimeError("Resolved backend changed; create a fresh run")
        binding = self.path / "data_manifest" / "binding.json"
        if binding.exists():
            saved = read_json(binding)
            manifest = self.path / "data_manifest" / "dataset.json"
            if not manifest.is_file() or sha256_file(manifest) != saved["manifest_sha256"]:
                raise RuntimeError("Bound dataset manifest is corrupt")
        if data is not None and self.data_identity != data.identity:
            raise RuntimeError("Dataset identity differs from this run")
        if data is not None and hasattr(data, "verify"):
            data.verify()
        return True

    def bind_data(self, identity, manifest_path):
        with self.writer_lock():
            self.verify()
            binding = self.path / "data_manifest" / "binding.json"
            if binding.exists():
                if read_json(binding)["identity"] != identity:
                    raise RuntimeError("Dataset changed; create a fresh run")
                if sha256_file(manifest_path) != read_json(binding)["manifest_sha256"]:
                    raise RuntimeError("Dataset manifest changed")
                return
            target = self.path / "data_manifest" / "dataset.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest_path, target)
            atomic_json(binding, {"identity": identity, "manifest_sha256": sha256_file(target)})

    def update_fit(self, name, status):
        """Caller holds writer_lock for the complete operation; preserve other fits."""
        self.fit_dir(name)  # validate names before changing the manifest
        p = self.path / "fits.json"
        current = read_json(p) if p.exists() else {"models": {}}
        current["models"][name] = status
        atomic_json(p, current)


def resolve_run(root, config, run_dir=None, use_saved_config=True):
    """None deliberately creates a new run. A supplied path resumes exactly that run."""
    root = Path(root).resolve()
    if run_dir is not None:
        supplied = Path(run_dir).expanduser()
        path = (supplied if supplied.is_absolute() else root / "runs" / supplied).resolve()
        if not path.is_dir() or path.parent != (root / "runs").resolve():
            raise ValueError("Select an existing immediate child of this project runs/ directory")
        with filesystem_lock(path / ".writer.lock"):
            saved = ExperimentConfig.from_dict(read_json(path / "config.json"))
            if not use_saved_config and canonical_hash(config.to_dict()) != canonical_hash(
                saved.to_dict()
            ):
                raise RuntimeError(
                    "Notebook settings differ (including max_epochs); use a fresh run"
                )
            selected = saved if use_saved_config else config
            result = Run(root, path, selected, select_device(), read_json(path / "identity.json"))
            result.verify()
        print(
            f"Resuming {path.name}; saved settings={use_saved_config}; hash={result.identity['config_hash']}"
        )
        return result
    config.validate()
    parent = root / "runs"
    parent.mkdir(exist_ok=True)
    device = select_device()
    source, env = source_manifest(root), environment_manifest()
    with filesystem_lock(parent / ".allocation.lock"):
        numbers = [
            int(p.name[4:])
            for p in parent.iterdir()
            if p.is_dir() and re.fullmatch(r"run_\d+", p.name)
        ]
        path = parent / f"run_{max(numbers, default=0) + 1:03d}"
        path.mkdir(exist_ok=False)
        identity = {
            "run_uuid": uuid.uuid4().hex,
            "protocol": config.protocol,
            "config_hash": canonical_hash(config.to_dict()),
            "source_hash": canonical_hash(source),
            "environment_hash": canonical_hash(env),
            "device": str(device),
            "created_at": datetime.now(UTC).isoformat(),
        }
        for rel in source:
            destination = path / "source_snapshot" / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / rel, destination)
        atomic_json(path / "config.json", config.to_dict())
        atomic_json(path / "environment.json", env)
        atomic_json(path / "source_manifest.json", source)
        atomic_json(path / "identity.json", identity)
        atomic_json(path / "fits.json", {"models": {}})
    print(f"Created {path.name}; fresh weights; hash={identity['config_hash']}")
    return Run(root, path, config, device, identity)
