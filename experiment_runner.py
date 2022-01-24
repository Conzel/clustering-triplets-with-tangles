#!/usr/bin/env python3
# Allows us to import tangles modules
import copy
import datetime
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

from baselines import Baseline
from data_generation import (generate_gmm_data_draw_means,
                             generate_gmm_data_fixed_means)
from plotting import AltairPlotter
from questionnaire import Questionnaire
from tangles.cost_functions import BipartitionSimilarity
from tangles.data_types import Cuts
from tangles.plotting import plot_hard_predictions
from tangles.tree_tangles import (ContractedTangleTree,
                                  compute_soft_predictions_children,
                                  tangle_computation)
from tangles.utils import (compute_cost_and_order_cuts,
                           compute_hard_predictions, normalize)

plt.style.use('ggplot')


class Configuration():
    def __init__(self, n, n_runs, seed, means, std, agreement,
                 name, num_distance_function_samples, noise, density,
                 redraw_means, min_cluster_dist, dimension,
                 n_components,
                 base_folder="results",
                 imputation_method="random",
                 baseline="none"):
        self.n = n
        self.seed = seed
        self.means = means
        self.std = std
        self.agreement = agreement
        self.name = name
        self.base_folder = base_folder
        self.num_distance_function_samples = num_distance_function_samples
        self.noise = noise
        self.density = density
        self.n_runs = n_runs
        self.redraw_means = redraw_means
        self.min_cluster_dist = min_cluster_dist
        self.dimension = dimension
        self.n_components = n_components
        self.imputation_method = imputation_method
        self.baseline = baseline

    def from_yaml(yaml_dict):
        return Configuration(**yaml_dict)

    def __str__(self) -> str:
        return "Configuration: " + str(self.__dict__)


class HardClusteringEvaluation():
    def __init__(self, y, y_pred) -> None:
        self.nmi = normalized_mutual_info_score(y, y_pred)
        self.ars = adjusted_rand_score(y, y_pred)
        self.n_clusters = np.unique(y_pred).size
        self.goal_clusters = np.unique(y).size


class RunResult():
    """
    A run result represents the results of a single run of the experiment.
    It is uniquely identified by the current time, results of the run, and configuration
    parameters that were used to generate the data.

    It contains the following fields:
        run_no, time_stamp, name, nmi, ars, no_clusters_found, agreement, min_cluster_dist, 
        std, n_components, dimension, noise, density, imputation, baseline

    The run result can directly be used as a row in a pandas dataframe.

    kind: str, 
        The kind of evaluation this contains. This can be 'baseline', or 'normal'.
    """

    def __init__(self, number: int, kind: str, config: Configuration, evaluation: HardClusteringEvaluation) -> None:
        self.run_no = int(number)
        self.agreement = config.agreement
        self.kind = kind
        self.nmi = evaluation.nmi
        self.ars = evaluation.ars
        self.no_found_clusters = evaluation.n_clusters
        self.n_components = config.n_components
        self.dimension = config.dimension
        self.noise = config.noise
        self.density = config.density
        self.imputation = config.imputation_method
        self.time_stamp = str(datetime.datetime.now())
        self.name = config.name
        self.min_cluster_dist = config.min_cluster_dist
        self.std = config.std

    def attributes() -> list:
        """
        Returns list of the attributes that characterize the experiments. 
        Results are not included.
        """
        return ['agreement', 'kind', 'n_components', 'dimension', 'noise', 'density', 'imputation', 'min_cluster_dist', 'std']

    def to_row(self) -> dict:
        """
        Returns a dictionary of itself that can be used as a row in a pandas dataframe.
        """
        return vars(self)

    def append_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds this run result to a dataframe.
        ! Returns a new df and doesn't change the old one.
        """
        return df.append(self.to_row(), ignore_index=True)


def run_experiment(conf: Configuration, workers=1) -> pd.DataFrame:
    """
    Runs an experiment with the given configuration. 

    In this example, we generate a synthetic dataset consisting of 
    n d-dimensional Gaussians. We then create a triplet questionnaire 
    of the datapoints and use this to create bipartitions.

    We then create a hard clustering using tangles.
    We plot the clustering and evaluate the clustering quality using NMI and ARS.

    By passing in a workers argument, we can do multiple runs of the experiment in parallel,
    greatly speeding up the process (on modern machines > 2x).
    If you want to use multiple processing, we advise passing in an argument of None 
    (which will use all available cores). Multiprocessing messes 
    up the console outputs, so you sadly won't see any progress on the single experiments.

    Returns a dataframe with the results of the runs.
    """
    seed = conf.seed

    if workers == 1:
        runner = _run_once_verbose
    else:
        runner = _run_once_quiet

    configs = []

    # Generating the configurations with the different seeds
    for i in range(conf.n_runs):
        backup_conf = copy.deepcopy(conf)
        backup_conf.seed = seed + i
        configs.append(backup_conf)

    # We need to pack the arguments to pass to the run function
    # as tuples, as imap doesn't like multiple arguments.
    args_to_runner = enumerate(configs)

    # For multiprocessing
    if workers is None or workers > 1:
        print("Running parallel experiments...")
        with Pool(workers) as pool:
            run_results = list(
                tqdm(pool.imap(runner, args_to_runner), total=len(configs)))
    else:
        run_results = list(map(runner, args_to_runner))

    df = pd.concat(run_results)
    df.to_csv(os.path.join(
        conf.base_folder, conf.name, conf.name + "_metrics.csv"))
    return df

# These two functions are defined because pool can only pickle top level functions (not lambdas)


def _run_once_verbose(conf_run_no_tuple: "tuple[Configuration, int]") -> "RunResult":
    run_no, conf = conf_run_no_tuple
    return _run_once(conf, run_no)


def _run_once_quiet(conf_run_no_tuple: "tuple[Configuration, int]") -> "RunResult":
    run_no, conf = conf_run_no_tuple
    return _run_once(conf, run_no, verbose=False)


def tangles_hard_predict(questionnaire: np.ndarray, agreement: int,
                         distance_function_samples=None, verbose=True) -> np.ndarray:
    """
    Uses the tangles algorithm to produce a hard clustering on the given data.

    questionnaire: np.ndarray of dimension (n_datapoints, n_questions),
        where each entry represents a triplet question of the form (a, (b,c))?
        which is set to true if a is closer to b than to c. 
        a is the datapoint in the row, (b,c) are associated with a column each.
    agreement: agreement parameter of the tangles algorithm,
        this should be less than the smallest cluster size you expect
    distance_function_samples: int,
        number of samples to use for the distance function (monte carlo approximation). If set to None, uses all
        samples. Setting this to something other than None might increase the speed of the algorithm
        but decreases it's performance.

    Returns predicted y labels for the data points as np.ndarray of dimension (n_datapoints,).
    """

    # Interpreting the questionnaires as cuts and computing their costs
    bipartitions = Cuts((questionnaire == 1).T)
    cost_function = BipartitionSimilarity(bipartitions.values.T)
    cuts = compute_cost_and_order_cuts(
        bipartitions, cost_function, verbose=verbose)

    # Building the tree, contracting and calculating predictions
    tangles_tree = tangle_computation(cuts=cuts,
                                      agreement=agreement,
                                      # print nothing
                                      verbose=int(verbose)
                                      )

    contracted = ContractedTangleTree(tangles_tree)
    contracted.prune(2, verbose=verbose)

    contracted.calculate_setP()

    # soft predictions
    weight = np.exp(-normalize(cuts.costs))
    compute_soft_predictions_children(
        node=contracted.root, cuts=bipartitions, weight=weight, verbose=3)

    ys_predicted, _ = compute_hard_predictions(
        contracted, cuts=bipartitions, verbose=verbose)

    return ys_predicted


def _generate_data(conf: Configuration):
    """
    Generates a synthetic dataset (mixture of gaussians) with the given configuration.
    """
    if conf.redraw_means:
        data = generate_gmm_data_draw_means(
            n=conf.n, std=conf.std, seed=conf.seed,
            dimension=conf.dimension, min_cluster_dist=conf.min_cluster_dist,
            components=conf.n_components)
    else:
        data = generate_gmm_data_fixed_means(
            n=conf.n, means=np.array(conf.means), std=conf.std, seed=conf.seed)

    return data


def _run_once(conf: Configuration, run_no: int, verbose=True) -> pd.DataFrame:
    """Runs the experiment once with the given configuration. Ignores
       n_runs parameter.

       If verbose is set to true, prints out steps in the process such as data generation etc.

       Returns a dataframe that contains the information as specified in RunResult.
    """
    np.random.seed(conf.seed)
    data = _generate_data(conf)
    if conf.noise > 0 and conf.imputation_method is None:
        raise ValueError("No imputation method given for noisy data.")
    # Creating the questionnaire from the data
    questionnaire = Questionnaire.from_euclidean(
        data.xs, noise=conf.noise, density=conf.density, seed=conf.seed,
        verbose=verbose)
    if conf.imputation_method is not None:
        questionnaire = questionnaire.impute(conf.imputation_method)
    y_predicted = tangles_hard_predict(questionnaire.values, conf.agreement,
                                       distance_function_samples=conf.num_distance_function_samples, verbose=verbose)

    # evaluate hard predictions
    assert data.ys is not None
    evaluation = HardClusteringEvaluation(data.ys, y_predicted)

    # save to df
    df = pd.DataFrame()
    df = RunResult(run_no, "normal", conf, evaluation).append_to_df(df)

    # Writing back results
    # Creating results folder if it doesn't exist
    result_output_path = Path(os.path.join(conf.base_folder, conf.name))
    result_output_path.mkdir(parents=True, exist_ok=True)
    if conf.dimension == 2:
        # Plotting the hard clustering
        plot_hard_predictions(data=data, ys_predicted=y_predicted,
                              path=result_output_path)

    # --- Checking if we need to calculate a baseline as well ---
    baseline_evaluation = None
    if conf.baseline is not None and conf.baseline.lower() != "none":
        if verbose:
            print("Evaluating baseline...")
        baseline = Baseline(conf.baseline)
        baseline_prediction = baseline.predict(
            data.xs, questionnaire, conf.n_components)
        baseline_evaluation = HardClusteringEvaluation(
            data.ys, baseline_prediction)
        df = RunResult(run_no, "baseline", conf,
                       baseline_evaluation).append_to_df(df)

    return df


def parameter_variation(parameter_values, name, attribute_name, base_config, plot=True, logx=False,
                        workers=1) -> pd.DataFrame:
    """
    Runs multiple experiments varying the given parameter. The results depicted in a 
    plot (with x = parameter values, y = nmi/ars). They are also saved in a csv file.
    The output is found under the base_config name + parameter_variation

    parameter_values: list of values the parameter can take
    name: name of the parameter (is used to save the folder and to label the plot)
    attribute_name: name of the attribute of the configuration to change
    base_config: Configuration object where we vary the parameter
    plot: Set to true if plots should be saved
    logx: Determines if the parameter value should have a logarithmic scale in the plot.

    Returns a pandas Dataframe containing the results, with rows containing
    the information specified in RunResult.
    """
    df = pd.DataFrame()
    seed = base_config.seed
    base_folder = os.path.join(
        "results", f"{base_config.name}-{name}_variation")

    for p in parameter_values:
        print(f"Calculating for {name} variation, value: {p}")
        conf = copy.deepcopy(base_config)
        if not hasattr(conf, attribute_name):
            raise ValueError(f"{attribute_name} not found in {conf}")
        setattr(conf, attribute_name, p)
        conf.name = f"{name}-{p:.4f}"
        conf.base_folder = base_folder

        df = df.append(run_experiment(conf, workers=workers))

    df.to_csv(os.path.join(base_folder, f"{name}_variation_results.csv"))
    if plot:
        plotter = AltairPlotter(base_folder)
        plotter.parameter_variation(df, attribute_name)
        plotter.save(f"{name}_variation_results.html")

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("First argument has to be the name of a YAML configuration. Exiting.")
        exit(1)
    # Loading the configuration
    with open(sys.argv[1], "r") as f:
        conf = Configuration.from_yaml(yaml.safe_load(f))

    if len(sys.argv) > 2 and (sys.argv[2] == "-p" or sys.argv[2] == "--parallel"):
        workers = None
    else:
        workers = 1

    # Running the experiment
    run_experiment(conf, workers=workers)
