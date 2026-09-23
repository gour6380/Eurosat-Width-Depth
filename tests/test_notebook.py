"""Runtime notebook structure tests permit owner-edited configuration and saved outputs."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_one_named_operation_per_cell_and_joint_views():
    notebook = json.loads((ROOT / "notebooks/eurosat_width_depth.ipynb").read_text())
    names = {key: [] for key in ("train_fit", "evaluate_fit", "benchmark_fit")}
    text = ""
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert index and notebook["cells"][index - 1]["cell_type"] == "markdown"
        source = "".join(cell["source"])
        text += source + "\n"
        tree = ast.parse(source)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
        ]
        assert len(calls) <= 1
        for call in calls:
            names[call.func.id].append(call.args[2].value)
    for action in names.values():
        assert action == ["shallow_wide", "middle", "deep_narrow"]
    for view in (
        "plot_learning_curves(run)",
        "plot_epoch_costs(run)",
        "plot_memory(run)",
        "quality_table(run)",
        "plot_confusions(run)",
    ):
        assert view in text
    assert "program-trace-width-depth" not in text
    assert "torch.load(" not in text


def test_optional_calls_are_guarded():
    notebook = json.loads((ROOT / "notebooks/eurosat_width_depth.ipynb").read_text())
    expected = {
        "train_fit": {"RUN_TRAINING"},
        "evaluate_fit": {"RUN_TEST_EVALUATION"},
        "benchmark_fit": {"RUN_BENCHMARKS"},
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in expected
            ):
                continue
            guards = set()
            node = call
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.If, ast.IfExp)):
                    guards.update(n.id for n in ast.walk(node.test) if isinstance(n, ast.Name))
            assert expected[call.func.id] <= guards


def test_active_notebook_has_no_pilot_or_gpu_monitoring_api():
    notebook = json.loads((ROOT / "notebooks/eurosat_width_depth.ipynb").read_text())
    removed_names = {
        "PilotConfig",
        "MonitoringConfig",
        "run_pilot",
        "pilot_table",
        "estimate_training_time",
        "gpu_monitor_status",
        "gpu_utilization_table",
        "plot_gpu_utilization",
        "RUN_PILOTS",
        "PILOT_VARIANTS",
        "RUN_MAIN_TRAINING",
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        # Saved outputs and markdown are historical context, not executable configuration.
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in removed_names
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                assert not removed_names.intersection(alias.name for alias in node.names)
                if isinstance(node, ast.ImportFrom):
                    assert node.module not in {"src.gpu_monitor", "src.gpu_reporting"}
