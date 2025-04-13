import random
import numpy as np
from evaluator import Evaluator
from data import get_dataset, get_synthetic_data
import argparse
import os
import shutil
import yaml
import logging
from datetime import datetime
import pandas as pd
from evolution import genetic_algorithm_selection
from baseline import random_sampling_parallel, best_from_original_synthetic_datasets


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    assert "dataset" in config, "Missing dataset in config"
    assert "experiment" in config, "Missing experiment in config"
    assert "genetic_algorithm" in config, "Missing genetic_algorithm in config"
    return config


def create_experiment_folder(folder_path):
    if not os.path.exists(rf'experiments\{folder_path}'):
        os.makedirs(rf'experiments\{folder_path}')

    if not os.path.exists(rf'experiments\{folder_path}\synthetic_data'):
        os.makedirs(rf'experiments\{folder_path}\synthetic_data')


def run_experiment(config, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    logging.info(f'Set random seed to {random_seed}.')

    create_experiment_folder(config['experiment'])
    folder_path = rf'experiments/{config["experiment"]}/'
    folder_path_synth = rf'experiments/{config["experiment"]}/synthetic_data/'
    genetic_alg_params = config['genetic_algorithm'][0]

    # load dataset
    logging.info(f'Loading dataset: {config["dataset"]}')
    dataset = get_dataset(config['dataset'])
    df_synth = get_synthetic_data(config['dataset'])
    n_sample = len(dataset.df_train)
    logging.info(f'Dataset: {config["dataset"]} loaded')

    logging.info(f'Task: Select the best k={len(dataset.df_train)} out of n={len(df_synth)} datapoints.')

    # evaluator
    evaluator = Evaluator(dataset)
    for func, args in zip(config["eval_funcs"], config["eval_funcs_params"]):
        if args != 'None':
            evaluator.register_evaluation_function(func, **args)
        else:
            evaluator.register_evaluation_function(func)

    logging.info(
        f'Evaluator with the following evaluation functions has been prepared: {", ".join(config["eval_funcs"])}.')

    # baselines
    logging.info(f'Computing baselines...')
    df_best_orig, best_orig_score = best_from_original_synthetic_datasets(df_synth, n_sample, evaluator)
    logging.info(f'Finished computing baseline: Best original synthetic dataset with score: {best_orig_score:.6f}.')
    df_best_orig.to_csv(folder_path_synth + 'best_orig.csv', index=False)
    logging.info(f'Best original synthetic dataset has been saved to "{folder_path_synth + "best_orig.csv"}"')

    tries = genetic_alg_params["population_size"] * (genetic_alg_params["generations"] + 1)
    print_every = genetic_alg_params["population_size"]

    logging.info(f'Computing Random Sampling baselines with {tries} repeats...')
    df_best_rand, best_rand_score = random_sampling_parallel(df_synth, n_sample, tries=tries, print_every=print_every,
                                                             evaluator=evaluator,
                                                             log_csv_file=folder_path + 'log_rand.csv')
    logging.info(f'Finished computing baseline: Best random synthetic dataset with score: {best_rand_score:.6f}.')
    df_best_rand.to_csv(folder_path_synth + 'best_rand.csv', index=False)
    logging.info(f'Best random synthetic dataset has been saved to "{folder_path_synth + "best_rand.csv"}"')

    # genetic alg
    logging.info(f'Computing Genetic Algorithm with random initialization...')
    best_gen_rand, best_gen_score = genetic_algorithm_selection(
        df_synth,
        subset_size=n_sample,
        population_size=genetic_alg_params["population_size"],
        generations=genetic_alg_params["generations"],
        tournament_size=genetic_alg_params["tournament_size"],
        mutation_rate=genetic_alg_params["mutation_rate"],
        num_mutations_fract=genetic_alg_params["num_mutations_fract"],
        initialization_strategy='random',
        evaluator=evaluator,
        log_csv_file=folder_path + 'log_gen_rand.csv'
    )
    logging.info(f'Finished Genetic Algorithm with random initialization with best score: {best_gen_score:.6f}.')

    best_gen_rand.to_csv(folder_path_synth + 'best_gen_rand.csv', index=False)
    logging.info(
        f'Best Genetic Algorithm with random initialization has been saved to "{folder_path_synth + "best_gen_rand.csv"}"')

    logging.info(f'Computing Genetic Algorithm with sequential initialization...')
    best_gen_rand, best_gen_score = genetic_algorithm_selection(
        df_synth,
        subset_size=n_sample,
        population_size=genetic_alg_params["population_size"],
        generations=genetic_alg_params["generations"],
        tournament_size=genetic_alg_params["tournament_size"],
        mutation_rate=genetic_alg_params["mutation_rate"],
        num_mutations_fract=genetic_alg_params["num_mutations_fract"],
        initialization_strategy='sequential',
        evaluator=evaluator,
        log_csv_file=folder_path + 'log_gen_seq.csv'
    )
    logging.info(f'Finished Genetic Algorithm with sequential initialization with best score: {best_gen_score:.6f}.')

    best_gen_rand.to_csv(folder_path_synth + 'best_gen_seq.csv', index=False)
    logging.info(
        f'Best Genetic Algorithm with random initialization has been saved to "{folder_path_synth + "best_gen_seq.csv"}"')

    # creating final file
    logging.info('Creating final experiment.csv...')
    df_rand = pd.read_csv(folder_path + 'log_rand.csv')
    df_gen_rand = pd.read_csv(folder_path + 'log_gen_rand.csv')
    df_gen_seq = pd.read_csv(folder_path + 'log_gen_seq.csv')

    df_final = pd.DataFrame()
    df_final['evaluations'] = df_rand['random_tries']
    df_final['generation'] = df_gen_rand['generation']
    df_final['baseline_random'] = df_rand['best']
    df_final['baseline_best'] = best_orig_score
    df_final['gen_rand'] = df_gen_rand['best_fitness']
    df_final['gen_seq'] = df_gen_seq['best_fitness']

    df_final.to_csv(folder_path + 'experiment.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description="Run genetic algorithm experiments based on config.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    create_experiment_folder(config["experiment"])

    # Save a copy of the config in the experiment folder for reproducibility.
    shutil.copy(args.config, os.path.join(rf'experiments\{config["experiment"]}', "config.yaml"))

    # Setup logging.
    log_file = os.path.join(rf'experiments\{config["experiment"]}', "experiment.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"Experiment started at {datetime.now()}")

    run_experiment(config)

    logging.info("Experiment finished.")


if __name__ == '__main__':
    main()
