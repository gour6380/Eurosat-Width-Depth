# Reproduction and evidence guide

## Read the result first

[The technical report](technical_report.md) and [curated evidence](../results/run_001/README.md) can be inspected without downloading data or creating a Python environment. `results/run_001` contains published records, not model weights or a resumable checkpoint directory.

The reference run used protocol `eurosat-width-depth-v3`, seed 42, 40 epochs per model, Python 3.13.15, PyTorch 2.13.0, and macOS on an M2 Pro with 16 GB unified memory. The complete recorded package environment is in [environment.json](../results/run_001/provenance/environment.json). Dependencies and source are preserved rather than upgraded during packaging.

## Environment

From the repository root:

```sh
uv sync --locked --python 3.13.15
.venv/bin/python -m ipykernel install --user --name eurosat-width-depth-py313 --display-name "EuroSAT Width–Depth — Python 3.13"
.venv/bin/jupyter lab
```

Open the notebook under `notebooks/` and select the named kernel. A new environment is required on a new machine; do not copy another machine's `.venv`. POSIX file locks make macOS/Linux the supported source targets; native Windows needs code changes. Only the recorded MPS run has been exercised here. No hidden CPU fallback is used when a selected accelerator operation fails.

## Notebook stages

The clean upload notebook keeps one editable configuration cell with the recorded 40-epoch recipe. `RUN_TRAINING`, `RUN_TEST_EVALUATION` and `RUN_BENCHMARKS` default to `False` in that copy. The original owner-executed notebook is preserved locally and may retain different display switches.

1. Run imports and review the configuration. Keep `RUN_DIR=None` to deliberately create a fresh run. Running its allocation cell again with `None` creates another run.
2. Run data preparation. It downloads EuroSAT RGB and the pinned spatial manifests, validates checksums/coverage, and creates a local cache. This requires network access on first preparation; later compatible runs reuse verified data.
3. Inspect training-only EDA and the parameter comparison.
4. Enable `RUN_TRAINING=True`, rerun the configuration cell, and execute the three training cells individually. Changing this display switch does not change the recipe. Do not rerun run allocation with `RUN_DIR=None` unless a fresh run is intended.
5. Inspect the saved learning curves, epoch timings, throughput and memory. Reporting functions read saved files; they do not train models.
6. Once all three fits finish, enable `RUN_TEST_EVALUATION=True`, rerun the configuration cell, then execute the freeze and evaluation cells. The freeze gate requires all three completed fits.
7. Enable `RUN_BENCHMARKS=True` only when ready to measure the selected models. Run the individual benchmark cells and joint comparison.
8. Assemble the report from saved results. It does not generate missing results automatically.

For a one-epoch smoke experiment, choose `max_epochs=1` before creating a new run. It is a different experiment and does not reproduce the reported results. A completed one-epoch run cannot be extended in place to 40 epochs.

## Resume deliberately

Before rerunning setup for an existing local experiment, set `RUN_DIR` to its printed name, for example `"run_002"`.

- `USE_SAVED_CONFIG=True`: the saved configuration is effective and is displayed.
- `USE_SAVED_CONFIG=False`: notebook and saved experiment settings must match.
- Incomplete fits resume from the last committed epoch. Partial epochs are restarted.
- Completed compatible fits are verified and reused without further optimizer updates.
- Source, dependency lock, complete installed package environment, Python/OS identity, backend, configuration and bound dataset must match. An identity failure is explicit; it is not permission to bypass checks.

The public configuration has the original experiment hash, but readers on another environment create a **new run identity**. Do not rename `results/run_001` into `runs/run_001`; weights and recovery payloads are intentionally excluded.

## Verification boundaries

The source and lockfile match the recorded source manifest. The public notebook differs only in publication presentation: cleared outputs/metadata, portable root discovery, explicit safe display switches and up-to-date explanatory text. These notebook changes do not alter `src`, models or the training recipe.

For a static integrity check of the upload copy, run:

```sh
python3 tools/verify_release.py
```

It uses the standard library, checks public evidence hashes and source identity, parses Python/notebook cells, and checks local Markdown links. It does not import `src`, instantiate models or run tests. The original `tools/check_static.py` is a historical clean-handoff checker and writes its own `docs/handoff_checks.json`; that receipt describes implementation checks, not the completed experiment.

The authored runtime tests can be run separately with `.venv/bin/python -m pytest`. They include synthetic fits and recovery exercises. They have **not** been executed as part of this packaging task, and no passing status is implied by the completed owner run. Real interrupted recovery remains untested by the published run, which contains one uninterrupted attempt per model.

## What gets uploaded

Upload the **contents** of the prepared `github-ready/eurosat-width-depth/` folder as the repository root. Include the hidden `.gitignore`. Its notebook is clean; the local executed notebook remains outside that upload copy. This folder is self-contained source plus evidence, without a runtime dependency on the original workspace.

Do not upload `.venv`, `data`, `runs`, notebook archives, logs or TensorBoard events. `.gitignore` prevents new untracked files from being selected by Git, but does not remove files that were already tracked. No Git commands, commits, pushes or publication are performed during this handoff.
