# Width vs Depth on EuroSAT

[Repository](https://github.com/gour6380/Eurosat-Width-Depth) · [Executed notebook](notebooks/eurosat_width_depth.ipynb) · [Technical report](docs/technical_report.md) · [Saved results](results/run_001/README.md)

**Does a deeper network help when the parameter budget stays almost the same?** This PyTorch experiment compares three residual CNNs on real satellite imagery, measuring classification quality, training cost and inference latency.

In the completed experiment, the **middle model reached 91.67% test accuracy in 14.57 minutes of recorded training/validation epoch time**. The deeper model took 30.86 minutes and reached 89.70%. This is an exploratory result from one seed, one spatial split and one fixed training recipe—not a universal ranking of width and depth.

![Test accuracy and recorded fit time for three parameter-matched CNNs](results/run_001/figures/accuracy_runtime.png)

## Results at a glance

All models trained for **40 epochs**, with seed **42**, on an **M2 Pro with 16 GB unified memory using PyTorch MPS**. Checkpoints were selected on validation data and frozen before test evaluation.

| Model | Blocks / stage | Stage widths | Parameters | Selected epoch | Test accuracy ↑ | Macro F1 ↑ | Recorded fit time ↓ |
|---|---:|---|---:|---:|---:|---:|---:|
| Shallow / wide | 1 | 48 / 96 / 192 | 691,834 | 34 | 89.81% | 0.8809 | 18.14 min |
| Middle | 2 | 32 / 64 / 128 | 696,618 | 34 | **91.67%** | **0.8984** | **14.57 min** |
| Deep / narrow | 4 | 22 / 44 / 88 | 697,344 | 28 | 89.70% | 0.8771 | 30.86 min |

Parameter spread is **0.80%**. Macro F1 weights each of the ten classes equally; accuracy weights each test image equally. Fit time sums all 40 recorded epoch totals, including validation and checkpoint/logging boundaries. It excludes notebook idle time and final timing-receipt/table/TensorBoard-series publication. There are no seed uncertainty bars.

The middle model used **19.7% less recorded fit time** than shallow/wide while improving test accuracy by **1.85 percentage points**. Shallow/wide had the lowest batch-1 forward latency. Deep/narrow had slightly higher selected validation accuracy than middle, but lower test accuracy; different class mixtures across the spatial partitions matter when interpreting that reversal.

**Read:** [technical report](docs/technical_report.md) · [protocol](docs/protocol.md) · [result tables and provenance](results/run_001/README.md) · [reproduction instructions](#reproduce-with-the-notebook)

## What this project implements

- Three CNN shapes with approximately equal parameter counts, using shared residual-block, normalization, pooling and classifier designs.
- Pinned, checksummed EuroSAT spatial partitions; normalization fitted on training pixels only.
- A notebook that calls reusable `src` modules, with one model operation per cell and joint comparison charts.
- Explicit numbered runs, saved architecture/configuration, atomic epoch checkpoints, writer locks and strict identity checks for resume.
- Validation-selected checkpoints, a test-access gate, per-class metrics, full saved test predictions and failure analysis.
- CSV/JSON as the evidence source, TensorBoard for monitoring, basic epoch timing and separate memory snapshots.

The completed run exercised training, checkpoint selection, test evaluation and inference measurements. **Standalone correctness/recovery tests remain unrun.** Successful training does not replace those tests; interrupted recovery was not exercised in this run.

## Dataset and experiment

Use **EuroSAT RGB**: 27,000 Sentinel-2 patches, native **64 × 64** RGB, ten land-cover classes. The published TorchGeo longitude-based partition gives **16,200 training / 5,400 validation / 5,400 test images**. It is pinned to an immutable manifest revision; it is not a newly generated random split. See [dataset attribution and sources](docs/attribution.md).

The fixed recipe is AdamW, learning rate `1e-3` with cosine decay to `1e-5`, weight decay `1e-4`, clipping `1.0`, batch `128`, validation batch `256`, FP32, and training-only horizontal/vertical flips. No pretrained weights, early stopping, HPO or extra seeds. The published notebook explicitly sets **40 epochs**; the library default of 20 is overridden.

Equal parameter counts do not equal equal compute, receptive fields or optimization difficulty. The spatial partition reduces random-image mixing but is not a guarantee of geographic independence. These results should not be compared directly with random-split EuroSAT leaderboard scores.

## Inspect without running models

The [curated evidence folder](results/run_001/README.md) contains seven PNG/SVG charts, aggregate tables, all test predictions, per-class metrics, benchmark requests/latencies, epoch records, selected-checkpoint hashes and exact configuration/data/environment identities. It contains no dataset pixels, model weights or training logs.

![Learning curves across the three shapes](results/run_001/figures/learning_curves.png)

Training accuracy approaches 99% while validation accuracy levels off. More training is not an established solution. The [failure analysis](docs/technical_report.md) examines class-specific errors and high-confidence mistakes without treating test failures as a tuning set.

Optional inference measurements are **synchronized FP32 model-forward timings**, with five warm-ups and 30 measured requests per case:

| Model | Batch 1 p50 ↓ | Batch 64 p50 ↓ |
|---|---:|---:|
| Shallow / wide | **2.28 ms** | 24.35 ms |
| Middle | 2.99 ms | **23.50 ms** |
| Deep / narrow | 4.69 ms | 32.13 ms |

These exclude image loading, normalization, transfer and output formatting. One sequential measurement block and a limited request panel do not establish a robust speedup or deployment latency. Memory values are separate snapshots, not peaks.

## Reproduce with the notebook

The published notebook includes **saved outputs from the completed 40-epoch run**. You can inspect the charts, tables and model results without executing it. These outputs are historical evidence; they do not supply the checkpoints or dataset needed to run the code again.

To run your own experiment:

1. Download the repository using **Code → Download ZIP**, or use this [download link](https://github.com/gour6380/Eurosat-Width-Depth/archive/refs/heads/main.zip).
2. Extract it and name the local project folder exactly **`eurosat-width-depth`**. The current notebook's root-discovery cell checks that folder name; the default ZIP folder name will not work unchanged.
3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed, then open a terminal in that project folder and recreate the locked Python 3.13 environment:

```sh
uv sync --locked --python 3.13.15
.venv/bin/python -m ipykernel install --user --name eurosat-width-depth-py313 --display-name "EuroSAT Width–Depth — Python 3.13"
.venv/bin/jupyter lab
```

Open [notebooks/eurosat_width_depth.ipynb](notebooks/eurosat_width_depth.ipynb) and select the named kernel. The saved notebook has training, test evaluation and benchmarking enabled from the completed run. **Review its configuration before using Run All.** For staged execution, begin with:

```python
RUN_DIR = None
USE_SAVED_CONFIG = True
RUN_TRAINING = False
RUN_TEST_EVALUATION = False
RUN_BENCHMARKS = False
```

Keep the explicit `TrainingConfig(seed=42, max_epochs=40, batch_size=128, validation_batch_size=256)` to use the reported training recipe. Run imports, configuration, run creation and data preparation in order. The preparation cell downloads the verified archive and split manifests if there is no compatible local cache.

Set `RUN_TRAINING=True` and rerun the configuration cell when ready, then execute the three model cells individually. After all three fits complete, enable test evaluation and run the freeze/evaluation cells. Benchmarking is a separate optional stage. Avoid rerunning the run-creation cell with `RUN_DIR=None` unless you intend to allocate another experiment.

`RUN_DIR=None` creates a new numbered run. Set `RUN_DIR="run_001"` (or the actual printed run name) to resume that local run later. `USE_SAVED_CONFIG=True` restores saved settings; `False` requires the notebook settings to match. Changing training settings, including the epoch limit, requires a fresh run. The published `results/run_001/` is an evidence bundle, **not** a resumable run.

The reference platform is macOS/Apple Silicon; the device preference is MPS → CUDA → CPU. Source uses POSIX file locks, so native Windows is not supported. CUDA/Linux and other backends have not been runtime-validated by this experiment. Exact numerical/timing reproduction on another platform is not promised. Resume also checks the recorded source, dependency lock, installed environment, backend and dataset identities; it is not a cross-machine checkpoint migration workflow.

## Repository layout

```text
src/                    reusable experiment code
notebooks/              modular notebook with completed-run outputs
tests/                  authored correctness and recovery tests
docs/                   protocol, technical report, reproduction and attribution
results/run_001/         saved metrics, figures and provenance
tools/                  static checks and release verification
pyproject.toml, uv.lock  exact dependencies and lockfile
data/, runs/            generated locally; ignored
```

Dataset caches, checkpoints, environments, TensorBoard events, logs, notebook backups and local packaging copies are excluded by `.gitignore`. The repository retains source, tests, reports, `uv.lock` and the curated `results/` evidence.

## Validation and remaining work

The [packaging check receipt](docs/publication_checks.json) records static and saved-artifact checks from release preparation, including a separate clean notebook export. The notebook published here retains its original executed outputs. This receipt is not a passing test-suite result. Run the saved-file integrity checker from the repository root without importing models:

```sh
python3 tools/verify_release.py
```

For the authored standalone correctness/recovery tests, run separately:

```sh
.venv/bin/python -m pytest
```

Next useful checks are those tests, visual inspection of selected failure images, and a more representative repeated inference benchmark if a deployment claim is needed. No additional training is required to inspect the published result.

Dataset and split credits are in [attribution.md](docs/attribution.md). No code license has been selected for this project; the EuroSAT dataset's terms are separate from the project's code licensing.
