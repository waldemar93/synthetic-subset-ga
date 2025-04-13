# Genetic Algorithm for Subset Selection in Synthetic Tabular Data

This repository implements a genetic algorithm to select better subsets from synthetic tabular datasets. Subsets can be optimized using any combination of evaluation metrics. Baseline methods such as random sampling and pre-generated subsets are included for comparison.

## Features

- Genetic algorithm with tournament selection, crossover, mutation, and elitism
- Flexible evaluation framework supporting statistical, predictive, and discriminative metrics
- Baseline comparisons with random sampling and best-of pre-generated subsets
- Parallelized fitness evaluation and structured experiment logging

## Installation

```bash
pip install -r requirements.txt
```

## Data Format

Datasets must be placed under `data/<dataset_name>/` with the following files:
- `train.csv`, `test.csv`: Real data splits
- `synthetic_data/*.csv`: Synthetic samples
- `columns.json`: Column metadata

Example `columns.json` for `heart_uci`:
```json
{
  "boolean": ["sex", "fbs", "exang"],
  "categorical": ["dataset", "cp", "restecg", "slope", "thal", "num"],
  "integer": ["age", "trestbps", "chol", "thalch", "ca"],
  "float": ["oldpeak"],
  "outcome": ["num"],
  "boolean_dtype": "int"
}
```

## Configuration

Experiments are configured via YAML. Example (`config.yaml`):

```yaml
dataset: 'heart_uci'
experiment: 'exp_heart_uci'
eval_funcs: ['log_corr', 'k_means', 'ml_cat', 'rsc', 'bsm', 'discriminator']
eval_funcs_params: [None, None, {metric: 'f1_micro'}, None, None, None]
genetic_algorithm:
  - population_size: 50
    generations: 100
    tournament_size: 5
    mutation_rate: 0.1
    num_mutations_fract: 0.02
```

## Running an Experiment

```bash
python run_experiment.py --config config.yaml
```

Results are saved under `experiments/<experiment_name>/`, including logs, selected subsets, and evaluation scores.

## Datasets

This project uses the following datasets (some preprocessed or subsampled):

- Stroke: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- Breast Cancer (BCW): https://archive.ics.uci.edu/dataset/15
- Heart Disease: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
- Thyroid Cancer: https://www.kaggle.com/datasets/mzohaibzeeshan/thyroid-cancer-risk-dataset
- ACTG Clinical Trial: https://github.com/sebp/scikit-survival/blob/master/sksurv/datasets/data/actg320.arff

