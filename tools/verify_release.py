"""Read-only publication checks: standard library only; no experiment execution."""

import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe publication path: {relative}")
    return path


def verify_manifest(root, manifest):
    records = json.loads(manifest.read_text())["files"]
    for relative, expected in records.items():
        if digest(checked_path(root, relative)) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    return len(records)


def local_links(document, root):
    count = 0
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
        target = target.strip().strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or not parsed.path:
            continue
        path = (document.parent / unquote(parsed.path)).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise ValueError(f"Broken local link in {document.relative_to(root)}: {target}")
        count += 1
    return count


def main():
    root = Path(__file__).resolve().parents[1]
    evidence = root / "results/run_001"
    verified = verify_manifest(evidence, evidence / "checksums.json")
    release = root / "release_manifest.json"
    if release.exists():
        verified += verify_manifest(root, release)

    source = json.loads((evidence / "provenance/source_manifest.json").read_text())
    for relative, expected in source.items():
        if digest(checked_path(root, relative)) != expected:
            raise ValueError(f"Source no longer matches the reported run: {relative}")

    paths = []
    for directory in ("src", "tests", "tools"):
        paths.extend(sorted((root / directory).rglob("*.py")))
    for path in paths:
        ast.parse(path.read_text(), filename=str(path.relative_to(root)))

    notebook = json.loads((root / "notebooks/eurosat_width_depth.ipynb").read_text())
    operations = {name: [] for name in ("train_fit", "evaluate_fit", "benchmark_fit")}
    code_cells = 0
    output_items = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        output_items += len(cell.get("outputs", []))
        if not index or notebook["cells"][index - 1]["cell_type"] != "markdown":
            raise ValueError(f"No explanation before notebook cell {index}")
        explanation = "".join(notebook["cells"][index - 1]["source"])
        if not all(f"**{key}:**" in explanation for key in ("Inputs", "Output", "Interpretation")):
            raise ValueError(f"Incomplete explanation before cell {index}")
        tree = ast.parse("".join(cell["source"]), filename=f"notebook cell {index}")
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in operations
        ]
        if len(calls) > 1:
            raise ValueError(f"Multiple fit operations in notebook cell {index}")
        for call in calls:
            operations[call.func.id].append(ast.literal_eval(call.args[2]))
        if release.exists() and (cell.get("execution_count") is not None or cell.get("outputs")):
            raise ValueError("Distribution notebook contains execution state")
    if any(names != ["shallow_wide", "middle", "deep_narrow"] for names in operations.values()):
        raise ValueError("Notebook fit matrix differs from the published comparison")

    documents = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
    documents += sorted(evidence.rglob("*.md"))
    links = sum(local_links(document, root) for document in documents)
    print(json.dumps({
        "scope": "Static and saved-file checks only; no project imports or execution",
        "verified_manifest_entries": verified,
        "source_identity_files": len(source),
        "python_files_parsed": len(paths),
        "notebook_code_cells": code_cells,
        "notebook_output_items": output_items,
        "local_markdown_links": links,
        "standalone_tests": "not run",
    }, indent=2))


if __name__ == "__main__":
    main()
