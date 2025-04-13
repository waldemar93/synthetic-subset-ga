import logging
import time
from concurrent.futures.process import ProcessPoolExecutor
from itertools import repeat
from typing import Literal, Optional, List
import numpy as np
import pandas as pd
from evaluator import Evaluator


def initialize_population_randomly(pop_size, total, subset_size) -> List:
    """
    Initialize the population with random candidate subsets.

    Parameters:
        pop_size: Number of candidate solutions.
        total: Total number of synthetic samples.
        subset_size: Number of samples to select for each candidate.

    Returns:
        A list of candidate solutions (each is a numpy array of indices).
    """
    population = []
    for _ in range(pop_size):
        candidate = np.random.choice(total, subset_size, replace=False)
        population.append(candidate)
    return population


def initialize_population_sequentially(total, subset_size):
    """
    Initialize the population with sequential candidate subsets.

    This creates subset_size individuals, where each candidate is a
    contiguous block of indices. The first candidate covers
    indices 0 to subset_size-1 and the last candidate covers indices
    total-subset_size to total-1.

    Parameters:
        total: Total number of synthetic samples.
        subset_size: Number of samples to select for each candidate.

    Returns:
        A list of candidate solutions (each is a numpy array of indices).
    """
    population = []

    init_pop_size = total // subset_size
    for i in range(init_pop_size):
        start = int(round(i * subset_size))
        candidate = np.arange(start, start + subset_size)
        population.append(candidate)

    return population


def fitness(candidate, synthetic_data, evaluator) -> float:
    """
    Compute the fitness for a candidate solution.

    Parameters:
        candidate: numpy array of selected indices.
        synthetic_data: The synthetic data array.
        evaluator: Evaluator used to evaluate the selected candidate solution.

    Returns:
        The fitness score (higher is better).
    """
    return evaluator.evaluate(synthetic_data.iloc[candidate])


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Select a candidate using tournament selection.

    Parameters:
        population: List of candidate solutions.
        fitnesses: List of fitness scores corresponding to the population.
        tournament_size: Number of candidates to consider in each tournament.

    Returns:
        A selected candidate from the population.
    """
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_index = indices[0]
    best_fit = fitnesses[best_index]
    for idx in indices:
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_index = idx
    return population[best_index]


def crossover(parent1, parent2, total, subset_size):
    """
    Combine two parent candidates to create a child candidate.

    The child is generated from the union of parent indices. If the union is too large, a random subset is chosen.

    Parameters:
        parent1: Parent candidate solutions (numpy arrays of indices).
        parent2: Parent candidate solutions (numpy arrays of indices).
        subset_size: Desired candidate size.

    Returns:
        A child candidate (numpy array of indices).
    """
    parent_union = set(parent1).union(set(parent2))
    child = list(parent_union)

    if len(child) > subset_size:
        child = np.random.choice(child, subset_size, replace=False).tolist()

    return np.array(child)


def mutate(candidate, total, mutation_rate=0.1, num_mutations=1):
    """
    Mutate a candidate solution with a given mutation rate.

    With probability `mutation_rate`, this function replaces `num_mutations`
    datapoints (indices) in the candidate with new indices not already in the candidate.

    Parameters:
        candidate: numpy array of candidate indices.
        total: Total number of synthetic samples.
        mutation_rate: Probability of applying a mutation.
        num_mutations: Number of indices to mutate when a mutation occurs.

    Returns:
        A mutated candidate (numpy array).
    """
    candidate = candidate.copy()
    if np.random.rand() < mutation_rate:
        # Determine the number of mutations to perform; ensure it does not exceed candidate size
        num_to_mutate = min(num_mutations, len(candidate))
        # Randomly select positions in the candidate to mutate
        indices_to_mutate = np.random.choice(len(candidate), num_to_mutate, replace=False)
        # Determine available indices that are not already in the candidate
        available = list(set(range(total)) - set(candidate))
        # In case there are fewer available indices than num_to_mutate
        num_available = min(len(available), num_to_mutate)
        if num_available > 0:
            new_indices = np.random.choice(available, num_available, replace=False)
            for pos, new_idx in zip(indices_to_mutate, new_indices):
                candidate[pos] = new_idx
    return candidate


# Top-level helper function for fitness evaluation.
def fitness_wrapper(candidate, synthetic_data, evaluator):
    return fitness(candidate, synthetic_data, evaluator)


def genetic_algorithm_selection(synthetic_data: pd.DataFrame, evaluator: Evaluator, subset_size=1000,
                                population_size=20, generations=50, tournament_size=3, mutation_rate=0.1,
                                num_mutations_fract=0.01,
                                initialization_strategy: Literal['random', 'sequential'] = 'random',
                                log_csv_file: Optional[str] = None):
    """
    Genetic Algorithm for subset selection.

    Parameters:
        synthetic_data: pd.DataFrame with all synthetic datapoints.
        evaluator: Evaluator with at least one registered evaluation function.
        subset_size: Number of synthetic samples to select.
        population_size: Number of candidate solutions in each generation.
        generations: Number of generations to run the algorithm.
        tournament_size: Size of tournament for selection.
        mutation_rate: Probability of mutation per candidate.
        num_mutations_fract: Fraction of the subset_size to mutate in each mutation event (0.01 -> 1% of the subset).
        initialization_strategy: Either 'random' (default) to initialize the population randomly or 'sequential' to
        initialize the population sequentially in chunks of subset_size. Note that in this case the population size
        is ignored before generation 1.
        log_csv_file: The filename (or path) of .csv file in which the information about the generations will be saved
    Returns:
        best_candidate: The candidate subset (indices) with the highest fitness.
        best_fitness: The corresponding fitness score.
    """
    # For experiment logging
    experiment_log = []
    start_time = time.time()

    total = synthetic_data.shape[0]
    num_mutations = round(num_mutations_fract * subset_size)

    print(f'Task: Select the best k={subset_size} out of n={total} datapoints.')

    if initialization_strategy == 'sequential':
        population = initialize_population_sequentially(total, subset_size)
    else:
        population = initialize_population_randomly(population_size, total, subset_size)

    start_calc_time = time.time()
    # execute fitness evaluation in parallel
    with ProcessPoolExecutor() as executor:
        fitnesses = list(executor.map(
            fitness_wrapper,
            population,
            repeat(synthetic_data),
            repeat(evaluator)
        ))

    logging.info(f'population calculation took {time.time() - start_calc_time} seconds.')
    best_candidate = population[np.argmax(fitnesses)]
    best_fitness = np.max(fitnesses)

    fitness_avg = np.average(fitnesses)
    # helpers for diversity evaluation
    coverage_ind = set()
    avg_equal_with_best = []

    # diversity evaluation
    for p in population:
        if not (p == best_candidate).all():
            avg_equal_with_best.append(len(set(p) & set(best_candidate)))
        coverage_ind.update(p)

    coverage_fraction = len(coverage_ind) / total
    average_overlap = np.average(avg_equal_with_best) / subset_size

    # Log initial generation metrics.
    experiment_log.append({
        'generation': 0,
        'best_fitness': best_fitness,
        'average_fitness': fitness_avg,
        'coverage_fraction': coverage_fraction,
        'average_overlap': average_overlap,
        'time_elapsed': time.time() - start_time
    })

    logging.info(f"Initial: Average Fitness = {fitness_avg:.6f}; Best Fitness = {best_fitness:.6f}; "
                 f"Coverage datapoints: {coverage_fraction * 100:.2f}%; average overlap with best: "
                 f"{average_overlap * 100:.2f}%")

    for gen in range(generations):
        new_population = []
        # helpers for diversity evaluation
        coverage_ind = set(best_candidate)
        avg_equal_with_best = []
        # Elitism: carry over the best candidate
        new_population.append(best_candidate)

        # Generate new candidates until we have a full population
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)

            # Exclude parent1 (and its fitness value) from the pool for parent2 selection.
            eligible_population = []
            eligible_fitnesses = []
            for cand, fit_val in zip(population, fitnesses):
                if not np.array_equal(cand, parent1):
                    eligible_population.append(cand)
                    eligible_fitnesses.append(fit_val)

            parent2 = tournament_selection(eligible_population, eligible_fitnesses, tournament_size)

            child = crossover(parent1, parent2, total, subset_size)
            child = mutate(child, total, mutation_rate, num_mutations)
            new_population.append(child)

            coverage_ind.update(child)
            avg_equal_with_best.append(len(set(child) & set(best_candidate)))

        population = new_population

        # execute fitness evaluation in parallel
        with ProcessPoolExecutor() as executor:
            fitnesses = list(executor.map(
                fitness_wrapper,
                population,
                repeat(synthetic_data),
                repeat(evaluator)
            ))

        fitness_avg = np.average(fitnesses)

        current_best_index = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_index]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_candidate = population[current_best_index]

        # diversity evaluation
        coverage_fraction = len(coverage_ind) / total
        average_overlap = np.average(avg_equal_with_best) / subset_size
        logging.info(f"Generation {gen+1}: Average Fitness = {fitness_avg:.6f}; Best Fitness = {best_fitness:.6f}; "
                     f"Coverage datapoints: {coverage_fraction * 100:.2f}%; average overlap with best: "
                     f"{average_overlap * 100:.2f}%")

        # Log generation metrics.
        experiment_log.append({
            'generation': gen+1,
            'best_fitness': best_fitness,
            'average_fitness': fitness_avg,
            'coverage_fraction': coverage_fraction,
            'average_overlap': average_overlap,
            'time_elapsed': time.time() - start_time
        })

    # Save log to CSV.
    if log_csv_file is not None:
        df_log = pd.DataFrame(experiment_log)
        df_log.to_csv(log_csv_file, index=False)
        logging.info(f"Experiment log saved to {log_csv_file}")

    return synthetic_data.iloc[best_candidate], best_fitness
