#!/usr/bin/env python3
# Allows us to import tangles modules
import sys
import os
from pathlib import Path
sys.path.append("./tangles")
# Otherwise the tangle tree algorithm may crash
sys.setrecursionlimit(5000)

# other imports
import pandas as pd
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import src.data_types as data_types
import src.utils as utils
import src.tree_tangles as tree_tangles
import src.cost_functions as cost_functions
import src.plotting as plotting
import sklearn
import yaml
from src.loading import load_GMM
from functools import partial
from questionnaire import generate_questionnaire, Questionnaire


class Configuration():
    def __init__(self, n, n_runs, seed, means, stds, agreement,
                 name, num_distance_function_samples, noise, density, base_folder="results"):
        self.n = n
        self.seed = seed
        self.means = means
        self.stds = stds
        self.agreement = agreement
        self.name = name
        self.base_folder = base_folder
        self.num_distance_function_samples = num_distance_function_samples
        self.noise = noise
        self.density = density
        self.n_runs = n_runs

    def from_yaml(yaml_dict):
        return Configuration(**yaml_dict)

    def __str__(self) -> str:
        return "Configuration: " + str(self.__dict__)


def run_experiment(conf: Configuration) -> "tuple[float, float]":
    """
    Runs an experiment with the given configuration. 

    In this example, we generate a synthetic dataset consisting of 
    n d-dimensional Gaussians. We then create a triplet questionnaire 
    of the datapoints and use this to create bipartitions.

    We then create a hard clustering using tangles.
    We plot the clustering and evaluate the clustering quality using NMI and ARS.

    Returns a tuple (ARS, NMI) of the resulting hard clustering.
    """
    seed = conf.seed
    ars_values = []
    nmi_values = []

    for i in range(conf.n_runs):
        conf_copy = copy.deepcopy(conf)
        conf_copy.seed = seed + i
        # Get resulting values
        ars, nmi = run_once(conf_copy)
        ars_values.append(ars)
        nmi_values.append(nmi)

    df = pd.DataFrame({"run": list(range(conf.n_runs)),
                      "ars": ars_values, "nmi": nmi_values})

    df.to_csv(os.path.join(conf.base_folder,
              conf.name, conf.name + "_metrics.csv"), index=False)
    return sum(ars_values) / len(ars_values), sum(nmi_values) / len(nmi_values)


def run_once(conf: Configuration) -> "tuple[float, float]":
    """Runs the experiment once with the given configuration. Ignores
       n_runs parameter.
    """
    # ---- loading parameters ----
    np.random.seed(conf.seed)

    num_clusters = len(conf.means)
    result_output_path = Path(os.path.join(conf.base_folder, conf.name))

    # Data generation
    xs, ys = load_GMM(blob_sizes=[conf.n] * num_clusters,
                      blob_centers=conf.means, blob_variances=conf.stds, seed=conf.seed)
    data = data_types.Data(xs=xs, ys=ys)

    # Creating the questionnaire from the data
    questionnaire = generate_questionnaire(
        data, noise=conf.noise, density=conf.density, seed=conf.seed).values

    # Interpreting the questionnaires as cuts and computing their costs
    bipartitions = data_types.Cuts((questionnaire == 1).T)
    cuts = utils.compute_cost_and_order_cuts(bipartitions, partial(
        cost_functions.mean_manhattan_distance, questionnaire, conf.num_distance_function_samples))

    # Building the tree, contracting and calculating predictions
    tangles_tree = tree_tangles.tangle_computation(cuts=cuts,
                                                   agreement=conf.agreement,
                                                   verbose=2  # print everything
                                                   )

    contracted = tree_tangles.ContractedTangleTree(tangles_tree)
    contracted.prune(5)

    contracted.calculate_setP()

    # soft predictions
    weight = np.exp(-utils.normalize(cuts.costs))
    tree_tangles.compute_soft_predictions_children(
        node=contracted.root, cuts=bipartitions, weight=weight, verbose=3)

    ys_predicted, _ = utils.compute_hard_predictions(
        contracted, cuts=bipartitions)

    # Creating results folder if it doesn't exist
    result_output_path.mkdir(parents=True, exist_ok=True)

    # evaluate hard predictions
    if data.ys is not None:
        ARS = sklearn.metrics.adjusted_rand_score(data.ys, ys_predicted)
        NMI = sklearn.metrics.normalized_mutual_info_score(
            data.ys, ys_predicted)
    else:
        raise ValueError("Data has no labels, not implemented yet.")

    # Plotting the hard clustering
    plotting.plot_hard_predictions(data=data, ys_predicted=ys_predicted,
                                   path=result_output_path)

    if data.ys is not None:
        return ARS, NMI


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("First argument has to be the name of a YAML configuration. Exiting.")
        exit(1)
    # Loading the configuration
    with open(sys.argv[1], "r") as f:
        conf = Configuration.from_yaml(yaml.safe_load(f))

    # Running the experiment
    run_experiment(conf)