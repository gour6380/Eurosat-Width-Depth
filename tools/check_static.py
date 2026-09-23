"""Static handoff checks only: parse files; never import src or execute notebook cells."""

import ast
import importlib.metadata
import json
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source_paths = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
for path in source_paths:
    ast.parse(path.read_text(), filename=str(path))
notebook = json.loads((root / "notebooks" / "eurosat_width_depth.ipynb").read_text())
operations = {"train_fit": [], "evaluate_fit": [], "benchmark_fit": []}
code_count = 0
for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    code_count += 1
    assert i and notebook["cells"][i - 1]["cell_type"] == "markdown", (
        f"Missing explanation before {i}"
    )
    explanation = "".join(notebook["cells"][i - 1]["source"])
    assert all(f"**{key}:**" in explanation for key in ("Inputs", "Output", "Interpretation"))
    assert cell["execution_count"] is None and cell["outputs"] == [], (
        "Distribution must be unexecuted"
    )
    tree = ast.parse("".join(cell["source"]), filename=f"notebook cell {i}")
    action_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in operations
    ]
    assert len(action_calls) <= 1, "Only one model operation per cell"
    for node in action_calls:
        assert isinstance(node.args[2], ast.Constant)
        operations[node.func.id].append(node.args[2].value)
for action, names in operations.items():
    assert names == ["shallow_wide", "middle", "deep_narrow"], (action, names)
assert notebook["metadata"]["kernelspec"]["name"] == "eurosat-width-depth-py313"
project = tomllib.loads((root / "pyproject.toml").read_text())
pins = {}
for requirement in project["project"]["dependencies"] + project["dependency-groups"]["dev"]:
    name, version = requirement.split("==")
    actual = importlib.metadata.version(name)
    assert actual == version, (name, version, actual)
    pins[name] = actual
lock = tomllib.loads((root / "uv.lock").read_text())
assert any(p["name"] == "eurosat-width-depth" for p in lock["package"])
counts = [(378 * n - 80) * b * b + (28 * n + 81) * b + 10 for n, b in [(1, 48), (2, 32), (4, 22)]]
assert counts == [691834, 696618, 697344]
assert (max(counts) - min(counts)) / min(counts) < 0.05
report = {
    "scope": "Static only; no src import, test execution, dataset or model execution",
    "source_and_test_files_parsed": len(source_paths),
    "notebook_code_cells": code_count,
    "notebook_outputs": 0,
    "one_model_action_per_cell": operations,
    "direct_pins": pins,
    "python": sys.version,
    "analytical_parameter_counts": counts,
    "runtime_tests": "authored, not executed",
    "dataset": "not executed this revision; existing owner-prepared cache preserved",
    "models": "not instantiated/trained/evaluated/benchmarked during this revision",
    "memory_fit": "revised execution path pending owner validation",
    "experimental_results": "pending",
}
(root / "docs" / "handoff_checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in report.items() if k not in ("direct_pins", "one_model_action_per_cell")},
        indent=2,
    )
)
