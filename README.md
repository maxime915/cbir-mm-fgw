# CBIR MM FGW

Implementation of 'Optimal Transport With Foundation Models for Multi-Stain CBIR in Digital Pathology' from ISBI2026.

This repository is an effort to improve reproducibility: all code here is sufficient to estimate the metrics reported in the paper.

This project is managed with [uv](https://docs.astral.sh/uv/getting-started/installation/), you can set up a new environment with the project installed by running `uv sync`.

Alternatively, you may install this project locally in editable mode using pip (`pip install -e .`) although the rest of the documentation assumes an installation with `uv`. If installed with pip, the scripts are available in the same environment as the package, you may simply omit the `uv run` in the following commands.

## Project structure

```bash
.
├── configs            # configurations files for the experiments
├── LICENSE
├── models-cache       # saved foundation models and correction modules
├── README.md
├── scripts            # convenience utils
├── src
│   └── cbir_mm_fgw
│       ├── hyreco.py  # everything related to the HyReCo dataset
│       ├── pipeline   # handling of foundation models and correction modules
│       └── run        # implementation of the experiments proposed in the paper
└── uv.lock
```

## Setup

The foundation models are hosted on HuggingFace and require authentication for download.
Login once before running any experiment:

```bash
uv run hf auth login
```

We provide a script to cache the models in `./models-cache`. You may set the environment variable `MODELS_CACHE` to the path of the requested directory if you wish to change this. We provide this pre-caching in a directory (different from HuggingFace's default) to work easier on a cluster, avoiding any modification of the models thus enabling concurrent jobs.

```bash
uv run cbir_mm_fgw_setup_model_cache
```

> [!NOTE]
> Model cache requires ~20 GB depending on selected models.

## Train a FGW correction module

The experiments are designed with [runexp](https://pypi.org/project/runexp/): each run uses a YAML config files, and results are stored in the `./.runexp` directory, along with the command that initiated them for reproducibility. We provide some useful config files in the `./configs` directory: use or tweak them as you need.

Training a FGW correction module is implemented in `./src/cbir_mm_fgw/run/train_fgw_core.py`, the command is available as follows. The models produced during this run will be saved in the model cache directory (`./models-cache/corrector_7-gwot-allpix-patches/`) with a unique name.

```bash
uv run cbir_mm_fgw_train_fgw configs/train_fgw_all.yaml
```

Note that there is a configuration option to enable the random pairing in the correction learning, to estimate the values from Table 3 in the article.

The FGW correction module uses _Fused Gromov Wasserstein_ optimal transport to extract paired patches from unpaired WSIs and learns a simple correction to map from the embeddings from the first WSI to the matching embeddings in the second WSI. Read the paper for more details.

## Train a z-score correction module

Training a z-score correction module is implemented in `./src/cbir_mm_fgw/run/train_zscore_corr_core.py`, the command is available as follows. The models produced during this run will be saved in the model cache directory (`./models-cache/corrector_z-score-corr/`) with a unique name.

```bash
uv run cbir_mm_fgw_train_zscore_corr configs/train_zscore_all.yaml
```

The z-score correction module estimate mean and variance for each component of the embedding of two domains and provide a mapping by shifting and rescaling embeddings of one domain to the other.

## Prepare search configuration files

If you wish to automatically create YAML configuration with your saved correction modules (recommended), we provide the script `./scripts/setup_search_config` which selects all saved corrections modules in the models cache and provide the results. This script outputs the part of the config related to the configuration to `stdout`: an example of its output is provided in `./configs/train_fgw_all.yaml`. NOTE: if you installed the project using pip instead of uv, replace `uv run` with the path of your python interpreter.

```bash
uv run scripts/setup_search_config.py
```

## Evaluate a search configuration

As for the training of the correction module, this functionality is implemented using _runexp_ in the file `./src/cbir_mm_fgw/run/search_core.py`. The command to execute it is given below. We provide examples in the directory `./configs`. You will find the accuracy metrics of your configuration in the subdirectory of `./.runexp/search` corresponding to your configuration.

```bash
uv run cbir_mm_fgw_search configs/train_fgw_all.yaml
```

## Assess parameter impact

Similarly, the impact of rotation and displacement can be studied in its own command with a dedicated configuration file.

```bash
uv run cbir_mm_fgw_assess_rot_disp configs/assess_rot_disp.yaml
```

## Evaluate OT pairing

We estimate the accuracy of the OT pairing process using the file `./src/cbir_mm_fgw/run/eval_ot_mapping_core.py` and its command below.

```bash
uv run cbir_mm_fgw_eval_ot_mapping configs/eval_ot_mapping.yaml
```

## Gotchas

### Model caching and reproducibility

We attempted to implement the foundation models in a reproducible way, such that anyone trying the same experiment would get very similar results (equal up to machine noise). The foundation model downloaded from HuggingFace are fixed with a revision and downloaded in the _model cache_ directory. Depending on the models you wish to download, HuggingFace may remove some downloaded artifacts from the cache. This is unfortunate as cache mutations prevent running multiple runs with different models on the same file system in parallel. This realization came too late. Our recommendation is to manually clear the cache, choose a limited selection of model and setup the cache for those models only then perform the experiment on those models, avoiding any modification of the cache.

## Using and citing CBIR MM FGW

> [!NOTE]
> This section will be updated after the conference proceedings are published.

```bibtex
@inproceedings{cbir_mm_fgw_2026,
  title     = {Optimal Transport With Foundation Models for Multi-Stain CBIR in Digital Pathology},
  author    = {Maxime Amodei and Raphaël Marée and Pierre Geurts},
  booktitle = {Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026},
  note      = {Accepted for publication}
}
```
