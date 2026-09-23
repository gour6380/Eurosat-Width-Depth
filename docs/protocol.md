# Registered EuroSAT width/depth protocol

Protocol v3; completed reference run recorded on 23 September 2026. The experiment uses real-world imagery, one seed and three fits. The owner completed 40 epochs per model, froze validation-selected checkpoints, evaluated the test set and recorded bounded inference measurements. See the [technical report](technical_report.md). Standalone correctness/recovery tests remain unrun; no interrupted recovery was exercised by this run.

## Data provenance and isolation

Use the creator-linked RGB archive at https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip. Its published MD5 is `f46e308c4d50d4bf32fedad2d3d62f3b`; preparation additionally records SHA-256. Images remain native 64×64 uint8 RGB. There are 27,000 patches and ten classes. No multispectral/TIFF data, model weights or learned tokenization are used.

Use TorchGeo's EuroSATSpatial manifests pinned at Hugging Face revision `1ce6f1bfb56db63fd91b6ecc466ea67f2509774c`, corresponding to the published longitude-based 60/20/20 split. Exact accepted filenames and SHA-256:

| Manifest | SHA-256 |
|---|---|
| eurosat-spatial-train.txt | 2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f |
| eurosat-spatial-val.txt | 6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b |
| eurosat-spatial-test.txt | de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452 |

The implementation checks full, disjoint coverage, duplicate identifiers, class support, native image dimensions and cache hashes. It does not invent a random replacement split on error. RGB mean/std use all training pixels after division by 255. EDA image selection and histograms are training-only; split/class counts may describe all partitions. Preparing and validating a cache is not model evaluation of its test partition.

Credit Patrick Helber, Benjamin Bischke, Andreas Dengel and Damian Borth, *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification* (2019). Dataset creator repository: https://github.com/phelber/EuroSAT. Preserve its MIT attribution and the applicable Copernicus Sentinel data notice. The project does not redistribute raw images in Git. Split provenance: https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/eurosat.py; implementation binds the immutable split revision rather than the mutable main branch.

The longitude split is stronger than newly mixing nearby patches at random. It is not claimed to provide a geographic buffer, scene-level independence or general deployment validity. Macro F1 exposes unequal class performance; actual counts and residual geographic limitations remain visible.

## Controlled model family

Three basic residual CNNs, with 1/2/4 two-convolution blocks per stage and base widths 48/32/22. All share three stages, stride-one 3×3 RGB stem, stride-two transitions into stages two/three, identity/projection shortcuts, bias-free convolutions, learned BatchNorm, ReLU, global average pooling and a ten-class linear head. No dropout, attention, transfer learning or special shape-specific recipes.

Count formula for `n` blocks per stage, base width `b`, and `c` classes:

`(378*n - 80)*b*b + (28*n + 41 + 4*c)*b + c`.

Default counts are 691,834, 696,618 and 697,344. Spread is `(max−min)/min = 0.7964%`; runtime rejects a spread above 5%. Learned normalization parameters count; running-statistic buffers do not. Main-path convolution depths are 7/13/25, each with two additional shortcut projections. Layers initialize independently. The comparison holds approximate capacity and the training recipe fixed, while receptive field and compute also change.

## Independent fixed training

Each training cell initializes, resumes or reuses exactly one named shape; there are no prerequisite fits. The reported comparison and public notebook use 40 epochs explicitly, overriding the library default of 20. Each model cell runs when RUN_TRAINING=True; cells execute sequentially. The clean upload notebook starts with training/evaluation/benchmark controls disabled. Choose the epoch limit before creating a run; changing it requires a fresh run.

One seed42, batch128, validation256, workers0 and FP32. AdamW starts at1e-3, cosine decays to1e-5, weight decay1e-4 and global norm clipping1.0. Horizontal/vertical flips operate per image on device. There is no early stopping, HPO, background activity collection or batch-level synchronized profiling. Basic epoch metrics, timings and memory snapshots remain.

Each fit has an independently seeded model and data-loader generator. Shared seeds align sample order, not differently shaped weights. MPS→CUDA→CPU is resolved once; unsupported operations, nonfinite losses and OOM fail explicitly, with no batch-size change or operation-level CPU fallback. Strict source/backend/environment identities govern resume.

Every epoch validates all validation images once. Highest validation accuracy selects the checkpoint; lower cross-entropy and earlier epoch break ties. Last-epoch recovery includes optimizer, scheduler, RNG, loader states and global step. A transaction commits immutable epoch payloads and an atomic manifest before final timing receipts. Missing final timing fields remain missing after interruption. No interrupted partial epoch becomes a completed result.

## Evaluation and operational evidence

Freeze all three main checkpoints before any test prediction. Report accuracy, macro F1, per-class support/precision/recall/F1, confusion counts, saved prediction IDs, confidence and mistakes. A training-majority class baseline supplies context. Confidence is uncalibrated, and diagnostic failures are not a basis for tuning against the same test set.

Report train/validation cross-entropy and accuracy, active epoch duration, training throughput and phase costs. Training accuracy uses augmentation; validation does not. Timers use backend synchronization only at epoch/phase boundaries. Final timing receipt/table/TensorBoard timing-series publication is outside the recorded total; incomplete timing coverage is explicit. Notebook idle time is not training time. Process RSS and available accelerator allocations are separate snapshots, never added or called peaks.

Optional inference benchmarking uses all three selected models, identical validation IDs, batch1/64, five warm-ups and30 requests. It measures synchronized FP32 forward time only, excluding image read/normalization/transfer and output formatting. Record requests, raw latencies, p50/p95, throughput and memory. One sequential block is a bounded measurement, not a robust paired speedup experiment.

Saved-only charts put all shapes together, use short display names, separate labels from legends and omit missing backend panels. Report assembly neither generates missing figures nor runs models. No cross-seed standard deviation or confidence bands are calculated from one seed.

## Deferred and unproven

No multispectral branch, pretrained backbone, Vision Transformer, extra seeds, search, arbitrary resolution change or deployment system in this first pass. The completed run provides execution evidence on the recorded MPS system. Standalone correctness tests, recovery under real interruption and visual failure-image review remain pending. Equal capacity, one recipe and one hardware platform cannot establish a universal width/depth rule.

## Compatibility and verification

Protocol v3 retains strict source, configuration, environment, backend and dataset identities. A fresh run initializes new weights and may reuse a verified local dataset cache. All three completed checkpoints are required before test evaluation. The public evidence folder is not a resumable run; checkpoints stay local.

GitHub preparation preserves src/, dependencies, the local executed notebook, original run and dataset. Public reports and a clean notebook copy were assembled from saved evidence; no retraining, test-suite execution, model evaluation or benchmarking was performed during packaging. CSV/JSON integrity and static checks do not establish numerical correctness or validate every recovery path. See the [reproduction guide](reproduce.md) for setup and verification boundaries.
