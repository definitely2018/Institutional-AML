# FIAD

This repository contains the implementation and experimental code for **FIAD**, a graph-distillation framework for communication-efficient cross-institutional anti-money laundering (AML) detection.

FIAD lets institutions condense their local transaction graphs into small synthetic graphs for one-shot sharing, followed by a short federated fine-tuning stage. The experiments compare FIAD with isolated learning, pooled-data learning, federated learning, and graph-compression baselines on the Elliptic and Elliptic++ Bitcoin transaction datasets.

## Repository structure

- `experiments`: main cross-institutional experiments.
- `models/`: graph neural network implementations.
- `data/`: dataset directory.
- `res/` : experimental outputs.


## Environment

The experiments were run with Python 3.10, PyTorch 2.0.1, and CUDA 11.8. A CUDA-capable GPU is recommended for the full experiments. Create the reference environment with:

```bash
conda env create -f environment.yml
conda activate fiad
```

## Data

Place the datasets under the following directories:

```text
data/elliptic_bitcoin_dataset/
data/Elliptic++ Dataset/
```

Dataset preprocessing is described in `data/README.md`.

## Running the experiments

Main experiment:

```bash
python cross_institution_experiment.py --dataset elliptic --seed 5 --reduction_rate 0.01 --gpu_id 0
```

Threshold-sensitivity analysis:

```bash
python threshold_sensitivity_original_protocol.py --dataset elliptic --seeds 5 15 25 35 45 --fl_rounds 50 --reduction_rate 0.01 --resume
```

Unified membership-inference analysis:

```bash
python unified_mia_experiment.py --datasets elliptic elliptic_pp --seeds 5 15 25 35 45 --resume
```

The scripts write detailed logs and result tables to `res/` or `results/`. Use `--help` to view the options supported by each script.

## Notes

This repository is intended for research reproduction. Some experiments require previously generated distilled-graph artifacts and can take substantial time and GPU memory to run.


