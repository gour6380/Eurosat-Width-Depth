# Dataset and split attribution

## EuroSAT RGB

Patrick Helber, Benjamin Bischke, Andreas Dengel and Damian Borth created EuroSAT. This experiment uses the RGB archive of Sentinel-2 patches, not the full multispectral version.

- [Creator repository and citation guidance](https://github.com/phelber/EuroSAT)
- [Zenodo dataset record and RGB archive checksum](https://zenodo.org/records/7711810)
- [2019 paper: EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://doi.org/10.1109/JSTARS.2019.2918242)
- [2018 paper: Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://doi.org/10.1109/IGARSS.2018.8519248)

The creator repository identifies the dataset as MIT-licensed and points users to the [Copernicus Sentinel data terms](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice). Follow the upstream notices when obtaining or redistributing imagery. This upload copy contains aggregate figures and prediction records, but no raw or processed dataset images.

## Spatial partition

The project uses the published TorchGeo EuroSATSpatial longitude-based train/validation/test lists, described in [TorchGeo's documentation](https://docs.torchgeo.org/en/stable/api/datasets/eurosat.html), rather than inventing a random split. The filenames are downloaded from [immutable dataset revision 1ce6f1bfb56db63fd91b6ecc466ea67f2509774c](https://huggingface.co/datasets/torchgeo/eurosat/tree/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c).

The exact archive, manifest and dataset-cache hashes are in [dataset.json](../results/run_001/provenance/dataset.json). Only published split manifests are used; TorchGeo itself is not a runtime dependency. The spatial split does not establish a geographic buffer or independence guarantee.

## Code and results

This repository is an independent experiment using the credited dataset and split protocol. It does not claim authorship of EuroSAT, TorchGeo or residual-network architecture concepts. Dataset licensing does not assign a license to this project's code. No project code license has been selected in this handoff.
