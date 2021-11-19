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
    def __init__(self, n, n_runs, seed, means, std, agreement,
                 name, num_distance_function_samples, noise, density,
                 redraw_means, min_cluster_dist, dimension,
                 n_components,
                 base_folder="results"):
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

    def from_yaml(yaml_dict):
        return Configuration(**yaml_dict)

    def __str__(self) -> str:
        return "Configuration: " + str(self.__dict__)


def draw_cluster_means(num_clusters, dimension, minimum_cluster_distance):
    """
    Draws cluster centers for the gaussian mixture model. The cluster centers
    are drawn according to a multidimensional gaussian. The cluster centers
    are then scaled so that the minimal distance between two clusters is equal
    to minimum_cluster_distance.

    Input:
        num_clusters: int,
            number of cluster centers
        dimension: int,
            dimension of the cluster centers
        minimum_cluster_distance: float,
            minimum distance that two cluster centers have 
    output: 
        ndarray of dimension [num_clusters, dimension],
        containing the cluster centers for each cluster
    """
    cluster_centers = np.random.randn(num_clusters, dimension)
    return rescale_points(cluster_centers, minimum_cluster_distance)


def rescale_points(x, desired_min_dist):
    """
    Scales an array x of points such that they have at least the given distance to each other.
    """
    if desired_min_dist > 0:
        dists = sklearn.metrics.pairwise_distances(x)
        current_min_dist = dists[~np.eye(dists.shape[0], dtype=bool)].min()
        return x / current_min_dist * desired_min_dist
    else:
        return x


def generate_gmm_data(conf: Configuration) -> data_types.Data:
    if conf.redraw_means:
        means = draw_cluster_means(conf.n_components, conf.dimension, conf.min_cluster_dist)
    else:
        means = conf.means
    stds = conf.std * np.ones(means.shape)
    num_clusters = len(means)
    assert num_clusters == conf.n_components

    xs, ys = load_GMM(blob_sizes=[conf.n] * num_clusters,
                      blob_centers=means, blob_variances=stds, seed=conf.seed)
    data = data_types.Data(xs=xs, ys=ys)
    return data


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
        backup_conf = copy.deepcopy(conf)
        backup_conf.seed = seed + i

        # Get resulting values
        ars, nmi = run_once(backup_conf)
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

    result_output_path = Path(os.path.join(conf.base_folder, conf.name))

    # ---- generating data ----
    data = generate_gmm_data(conf)

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

    if conf.dimension == 2:
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
