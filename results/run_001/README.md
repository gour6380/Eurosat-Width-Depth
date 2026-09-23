# Completed EuroSAT comparison: run_001

This is a compact evidence bundle from the completed 23 September 2026 experiment. **It is not a resumable run.** Model weights, image data, logs and the local environment are intentionally excluded.

The [technical report](../../docs/technical_report.md) explains methods, observations, failures and limitations. The middle model had the highest observed test accuracy (91.67%) and lowest recorded fit time (14.57 minutes). One seed and one recipe do not establish a universal architecture ranking.

## Start here

- [Result summary](tables/result_summary.csv): one row per model; exact metrics and recorded times.
- [Original quality table](tables/quality.csv): selected checkpoints, test accuracy, macro F1 and loss.
- [Epoch table](tables/epochs.csv): all 120 completed epochs, including timing coverage and memory snapshots.
- [Benchmark summary](tables/benchmark_summary.csv): batch-1/64 latency, throughput and separate memory snapshots.
- [Class counts](tables/class_counts.csv): unequal train/validation/test class support.
- [Technical report](../../docs/technical_report.md): analysis and figure interpretations.

## Charts

| Figure | What it shows |
|---|---|
| [Accuracy vs fit time](figures/accuracy_runtime.png) | Test accuracy against the sum of recorded epoch totals; higher accuracy and lower time are preferable. |
| [Learning curves](figures/learning_curves.png) | Training/validation accuracy and loss through 40 epochs; training uses augmentation. |
| [Held-out quality](figures/quality.png) | Accuracy and macro F1; both higher-is-better, with different weighting. |
| [Confusion matrices](figures/confusion_matrices.png) | Which true classes are confused with which predicted classes; support differs by class. Exact counts are in evaluation CSVs. |
| [Epoch costs](figures/epoch_costs.png) | Actual phase and whole-epoch measurements; no extrapolation from sampled batches. |
| [Throughput](figures/throughput.png) | Training examples per measured training second. |
| [Memory](figures/memory.png) | RSS, MPS live and driver allocations as distinct snapshots; never add them or call them peaks. |

Every chart also has an SVG with the same basename. Figures are byte-identical to the reviewed completed-run outputs; packaging did not redraw or rerun them.

## Evidence layout and definitions

- `provenance/`: exact configuration, run identity, installed environment, dataset/source hashes and frozen checkpoint selections.
- `training/<shape>/`: architecture description, parameter count, epoch records, attempt history and all completed timing receipts.
- `evaluation/<shape>/`: metrics, per-class scores, confusion counts, 5,400 saved test predictions and every incorrect prediction.
- `benchmarks/<shape>/`: identical validation-request IDs, all per-request forward latencies and summary definitions.
- `tables/`: original analysis tables plus two derived summaries.
- [Copied-artifact manifest](copied_artifacts.json): mapping to original local-run paths and byte-for-byte checksums.
- [Evidence checksums](checksums.json): all public evidence files except this checksum manifest itself.

Fractions in CSV/JSON are on a 0–1 scale unless a field explicitly says otherwise. Reports display accuracy as percentages. A `confidence` is the highest softmax probability, not a calibrated probability of correctness. Integer labels follow the ordered `class_names` in [dataset.json](provenance/dataset.json). Image IDs identify upstream samples; no pixels are distributed here.

Timing units are seconds in epoch/latency records and milliseconds in benchmark summaries. Epoch totals include training, validation, checkpointing and the recorded logging boundary; they exclude final receipt/table/TensorBoard timing-series publication. Inference is synchronized FP32 model-forward only. Read the [benchmark report section](../../docs/technical_report.md) before interpreting speed or memory.

References to `models/.../*.pt` inside copied JSON files are original provenance strings. Those weights are not included, and this folder cannot be passed to the notebook's resume interface. Checkpoint hashes allow verification if the owner separately supplies the original weights.

## Verification status

Saved metrics, prediction counts, selection records and timing aggregates were checked without executing models. The source and lockfile match the recorded source manifest. Standalone tests and interrupted-recovery validation remain pending; real training success does not replace them. Visual inspection of individual failure images is also pending.

The checksum manifests detect accidental changes; they are not cryptographic proof of authorship or independent scientific replication.
