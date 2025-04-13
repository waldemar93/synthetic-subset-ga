import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from evaluator import Evaluator


# Top-level helper function for one random try.
def random_try(i, df_synth, n_sample, evaluator):
    total_size = len(df_synth)
    # Generate candidate indices randomly.
    candidate = np.random.choice(total_size, n_sample, replace=False)
    # Get the candidate DataFrame.
    df_synth_candidate = df_synth.iloc[candidate]
    # Evaluate the candidate.
    result = evaluator.evaluate(df_synth_candidate)
    return i, result, candidate  # return the try index, result, and candidate indices


def random_sampling_parallel(df_synth, n_sample, tries, print_every=100, evaluator=None, log_csv_file=None):
    """
    Parallel version of random sampling.

    For each try, a candidate subset is drawn randomly and evaluated.
    Results are processed as tasks complete, and intermediate logging is done.

    Parameters:
        df_synth: The synthetic dataset as a DataFrame.
        n_sample: Number of samples to select for each candidate.
        tries: Total number of random sampling tries.
        print_every: How many completed tasks between log updates.
        evaluator: An object with an evaluate() method.
        log_csv_file: Optional file path to save the log CSV.

    Returns:
        best_df: The candidate DataFrame (subset of df_synth) that achieved the best result.
        best: The best evaluation result.
    """
    experiment_log = []
    results = []
    best_df = None
    best = -np.Infinity

    # Use a ProcessPoolExecutor to run random_try in parallel.
    with ProcessPoolExecutor() as executor:
        # Submit all tries to the executor.
        futures = {executor.submit(random_try, i, df_synth, n_sample, evaluator): i for i in range(tries)}

        completed = 0
        # Process tasks as they complete.
        for future in as_completed(futures):
            i, result, candidate = future.result()
            results.append(result)
            if result > best:
                best = result
                best_df = df_synth.iloc[candidate]
            completed += 1

            if completed % print_every == 0:
                avg_val = np.average(results)
                logging.info(f'Random tries: {completed}: avg={avg_val:.6f}; best={best:.6f}')
                experiment_log.append({
                    'random_tries': completed,
                    'best': best,
                    'average': avg_val
                })

    logging.info(f'Final: Random tries: {tries}: avg={np.average(results):.6f}; best={best:.6f}')

    # Save the experiment log if a file path is provided.
    if log_csv_file is not None:
        df_log = pd.DataFrame(experiment_log)
        df_log.to_csv(log_csv_file, index=False)
        logging.info(f"Experiment log saved to {log_csv_file}")

    return best_df, best


def random_sampling(df_synth, n_sample, tries, print_every=100, evaluator: Evaluator = None, log_csv_file=None):
    # For experiment logging
    experiment_log = []

    results = []
    best_df = None
    best = -np.Infinity
    for i in range(tries):
        # df_synth_candidate = df_synth.sample(n=n_sample, random_state=i)
        total_size = len(df_synth)
        candidate = np.random.choice(total_size, n_sample, replace=False)
        df_synth_candidate = df_synth.iloc[candidate]
        result = evaluator.evaluate(df_synth_candidate)

        results.append(result)
        if result > best:
            best = result
            best_df = df_synth_candidate

        if (i + 1) % print_every == 0:
            logging.info(f'Random tries: {i + 1}: avg={np.average(results):.6f}; best={best:.6f}')
            experiment_log.append({
                'random_tries': i + 1,
                'best': best,
                'average': np.average(results)
            })

    logging.info(f'final:\nRandom tries: {tries}: avg={np.average(results):.6f}; best={best:.6f}')

    # Save log to CSV.
    if log_csv_file is not None:
        df_log = pd.DataFrame(experiment_log)
        df_log.to_csv(log_csv_file, index=False)
        logging.info(f"Experiment log saved to {log_csv_file}")

    return best_df, best


def best_from_original_synthetic_datasets(df_synth, n_sample, evaluator: Evaluator = None):
    results = []
    best_df = None
    best = -np.Infinity

    num_datasets = len(df_synth) // n_sample
    for i in range(num_datasets):
        start = int(round(i * n_sample))
        dataset = df_synth.iloc[np.arange(start, start + n_sample)]
        result = evaluator.evaluate(dataset)
        results.append(result)

        if result > best:
            best = result
            best_df = dataset

    return best_df, best

