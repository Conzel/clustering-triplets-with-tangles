#!/usr/bin/env python3
"""
Experiments where we vary the parameters of the tangles in a questionnaire scenario
and visualize their results.
"""

import pandas as pd
import copy
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics
plt.style.use('ggplot')

from data_generation import Configuration, run_experiment


def parameter_variation(parameter_labels, parameter_values, name, attribute_name, base_config, logx=False):
    ars_values = []
    nmi_values = []
    seed = base_config.seed
    base_folder = os.path.join("results", f"06-{name}_variation")

    for l, p in zip(parameter_labels, parameter_values):
        print(f"Calculating for {name} variation, value: {l}")
        conf = copy.deepcopy(base_config)
        if not hasattr(conf, attribute_name):
            raise ValueError(f"{attribute_name} not found in {conf}")
        setattr(conf, attribute_name, p)
        conf.name = f"{name}-{l:.4f}"
        conf.base_folder = base_folder

        ars, nmi = run_experiment(conf)
        ars_values.append(ars)
        nmi_values.append(nmi)

    # Saving the results
    metric_results = {f"{name}": parameter_labels,
                      'nmi': nmi_values, 'ars': ars_values}
    df = pd.DataFrame(data=metric_results)
    df.to_csv(os.path.join(base_folder, "metric_results.txt"), index=False)

    # Plotting
    plt.figure()
    plt.plot(parameter_labels, ars_values, "--^", label="ARS")
    plt.plot(parameter_labels, nmi_values, "--o", label="NMI")
    if logx:
        plt.yscale("log")
    plt.title(f"{name} variation")
    plt.legend()
    plt.savefig(os.path.join(base_folder, f"{name}_variation.png"))


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
    dists = sklearn.metrics.pairwise_distances(cluster_centers)
    min_dist = dists[~np.eye(dists.shape[0], dtype=bool)].min()
    return cluster_centers / min_dist * minimum_cluster_distance


if __name__ == "__main__":
    base_config = Configuration.from_yaml(
        yaml.load(open("experiments/06-base-config.yaml")))
    # Varying the agreement parameter
    agreements = list(range(1, 21, 2))
    noise = np.arange(0, 1, 0.05)
    density = np.logspace(-3, 0, num=20)
    minimum_cluster_distances = np.arange(0.5, 5, 0.5)
    # drawing means
    means = list(map(lambda d: draw_cluster_means(
        5, 2, d), minimum_cluster_distances))

    # parameter_variation(agreements, agreements, "agreement", "agreement", base_config)
    # parameter_variation(noise, noise, "noise", "noise", base_config)
    # parameter_variation(density, density, "density", "density", base_config, logx=True)
    parameter_variation(minimum_cluster_distances, means,
                        "minimum_cluster_distance", "means", base_config)
