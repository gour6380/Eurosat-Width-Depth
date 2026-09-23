"""Lightweight run-identity and writer-exclusion acceptance tests."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.config import ExperimentConfig
from src.io import atomic_json, canonical_hash, read_json
from src.runs import resolve_run


class FakeDevice:
    def __init__(self, kind="cpu"):
        self.type = kind

    def __str__(self):
        return self.type


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    from src import runs

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "example.py").write_text("VALUE = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(runs, "environment_manifest", lambda: {"python": "test", "packages": {}})
    monkeypatch.setattr(runs, "select_device", FakeDevice)
    return tmp_path


def test_explicit_numbering_never_auto_reuses_matching_configuration(isolated_project):
    config = ExperimentConfig()
    first = resolve_run(isolated_project, config)
    second = resolve_run(isolated_project, config)
    assert first.path.name == "run_001"
    assert second.path.name == "run_002"
    assert first.identity["config_hash"] == second.identity["config_hash"]
    assert first.identity["run_uuid"] != second.identity["run_uuid"]
    assert first.verify()


def test_explicit_resume_and_saved_configuration(isolated_project):
    config = ExperimentConfig()
    created = resolve_run(isolated_project, config)
    edited = replace(config, training=replace(config.training, max_epochs=7))
    resumed = resolve_run(isolated_project, edited, created.path.name, use_saved_config=True)
    assert resumed.path == created.path
    assert resumed.identity == created.identity
    assert resumed.config.training.max_epochs == config.training.max_epochs
    with pytest.raises(RuntimeError, match="Notebook settings differ"):
        resolve_run(isolated_project, edited, created.path, use_saved_config=False)


def test_epoch_limit_is_part_of_canonical_identity():
    config = ExperimentConfig()
    saved = config.to_dict()
    restored = ExperimentConfig.from_dict(saved)
    assert canonical_hash(saved) == canonical_hash(restored.to_dict())
    changed = replace(config, training=replace(config.training, max_epochs=21))
    assert canonical_hash(config.to_dict()) != canonical_hash(changed.to_dict())
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_configuration_has_one_training_budget_and_no_optional_pilot_or_monitoring():
    config = ExperimentConfig()
    saved = config.to_dict()
    assert "max_epochs" in saved["training"]
    for removed in ("pilot", "monitoring"):
        assert removed not in saved
        assert not hasattr(config, removed)
    assert ExperimentConfig.from_dict(saved).to_dict() == saved


def test_foreign_project_path_is_rejected(isolated_project, tmp_path_factory):
    foreign = tmp_path_factory.mktemp("foreign")
    with pytest.raises(ValueError, match="immediate child"):
        resolve_run(isolated_project, ExperimentConfig(), foreign)


def test_writer_exclusion_blocks_second_writer_and_resume(isolated_project):
    run = resolve_run(isolated_project, ExperimentConfig())
    with run.writer_lock():
        with pytest.raises(RuntimeError):
            with run.writer_lock():
                pytest.fail("Second writer unexpectedly acquired the lock")
        with pytest.raises(RuntimeError):
            resolve_run(isolated_project, run.config, run.path)
    with run.writer_lock():
        pass


def test_cumulative_fit_manifest_preserves_preceding_work(isolated_project):
    run = resolve_run(isolated_project, ExperimentConfig())
    with run.writer_lock():
        run.update_fit("shallow_wide", {"state": "completed", "epoch": 20})
        run.update_fit("middle", {"state": "running", "epoch": 1})
        run.update_fit("middle", {"state": "completed", "epoch": 2})
    result = read_json(run.path / "fits.json")
    assert set(result) == {"models"}
    assert result["models"]["shallow_wide"]["epoch"] == 20
    assert result["models"]["middle"]["epoch"] == 2
    with pytest.raises(ValueError, match="Unregistered"):
        run.fit_dir("unknown")


@pytest.mark.parametrize("relative", ["config.json", "source_manifest.json", "environment.json"])
def test_corrupt_saved_provenance_blocks_resume(isolated_project, relative):
    run = resolve_run(isolated_project, ExperimentConfig())
    path = run.path / relative
    original = read_json(path)
    original["unexpected_corruption"] = True
    atomic_json(path, original)
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        resolve_run(isolated_project, run.config, run.path)
    assert len(list((isolated_project / "runs").glob("run_*"))) == 1


def test_corrupt_source_snapshot_and_current_source_are_rejected(isolated_project):
    run = resolve_run(isolated_project, ExperimentConfig())
    snapshot = run.path / "source_snapshot" / "src" / "example.py"
    snapshot.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="snapshot is corrupt"):
        run.verify()
    snapshot.write_text("VALUE = 1\n")
    (isolated_project / "src" / "example.py").write_text("VALUE = 3\n")
    with pytest.raises(RuntimeError, match="Source/dependency lock changed"):
        run.verify()


def test_environment_and_backend_changes_are_rejected(isolated_project, monkeypatch):
    from src import runs

    run = resolve_run(isolated_project, ExperimentConfig())
    monkeypatch.setattr(runs, "environment_manifest", lambda: {"python": "changed"})
    with pytest.raises(RuntimeError, match="Environment changed"):
        run.verify()
    monkeypatch.setattr(runs, "environment_manifest", lambda: {"python": "test", "packages": {}})
    monkeypatch.setattr(runs, "select_device", lambda: FakeDevice("mps"))
    with pytest.raises(RuntimeError, match="backend changed"):
        run.verify()


def test_dataset_binding_is_immutable_and_manifest_verified(isolated_project):
    run = resolve_run(isolated_project, ExperimentConfig())
    manifest = isolated_project / "dataset_manifest.json"
    atomic_json(manifest, {"dataset": "fixture", "files": {}})
    identity = canonical_hash(read_json(manifest))
    run.bind_data(identity, manifest)
    run.bind_data(identity, manifest)
    assert run.verify(SimpleNamespace(identity=identity))
    with pytest.raises(RuntimeError, match="Dataset changed"):
        run.bind_data("another-identity", manifest)
    with pytest.raises(RuntimeError, match="Dataset identity differs"):
        run.verify(SimpleNamespace(identity="wrong"))
    atomic_json(run.path / "data_manifest" / "dataset.json", {"changed": True})
    with pytest.raises(RuntimeError, match="dataset manifest is corrupt"):
        run.verify()


@pytest.mark.parametrize(
    "field,value", [("max_epochs", 1.5), ("batch_size", True), ("seed", True), ("seed", 2**32)]
)
def test_invalid_integral_training_settings_fail_before_run_allocation(field, value):
    config = ExperimentConfig()
    changed = replace(config, training=replace(config.training, **{field: value}))
    with pytest.raises(ValueError):
        changed.validate()
