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


def parameter_variation(parameter_values, name, attribute_name, base_config, logx=False):
    ars_values = []
    nmi_values = []
    seed = base_config.seed
    base_folder = os.path.join("results", f"06-{name}_variation")

    for p in parameter_values:
        print(f"Calculating for {name} variation, value: {p}")
        conf = copy.deepcopy(base_config)
        if not hasattr(conf, attribute_name):
            raise ValueError(f"{attribute_name} not found in {conf}")
        setattr(conf, attribute_name, p)
        conf.name = f"{name}-{p:.4f}"
        conf.base_folder = base_folder

        ars, nmi = run_experiment(conf)
        ars_values.append(ars)
        nmi_values.append(nmi)

    # Saving the results
    metric_results = {f"{name}": parameter_values,
                      'nmi': nmi_values, 'ars': ars_values}
    df = pd.DataFrame(data=metric_results)
    df.to_csv(os.path.join(base_folder, "metric_results.txt"), index=False)

    # Plotting
    plt.figure()
    plt.plot(parameter_values, ars_values, "--^", label="ARS")
    plt.plot(parameter_values, nmi_values, "--o", label="NMI")
    if logx:
        plt.xscale("log")
    plt.title(f"{name} variation")
    plt.legend()
    plt.savefig(os.path.join(base_folder, f"{name}_variation.png"))


if __name__ == "__main__":
    base_config = Configuration.from_yaml(
        yaml.load(open("experiments/06-base-config.yaml")))
    # Varying the agreement parameter
    agreements = list(range(1, 40, 1))
    noise = np.arange(0, 1, 0.05)
    density = np.logspace(-3, 0, num=20)
    minimum_cluster_distances = np.arange(0.5, 5, 0.5)
    stds = np.arange(0.1, 5.1, 0.5)
    dimensions = np.arange(1, 11, 1)
    n_components = np.arange(2, 11, 1)

    redraw_conf = copy.deepcopy(base_config)
    redraw_conf.redraw_means = True
    parameter_variation(agreements, "agreement", "agreement", redraw_conf)
    parameter_variation(noise, "noise", "noise", redraw_conf)
    parameter_variation(density, "density",
                        "density", redraw_conf, logx=True)
    parameter_variation(minimum_cluster_distances,
                        "min_cluster_dist", "min_cluster_dist", redraw_conf)
    parameter_variation(noise, "noise", "noise", redraw_conf)
    parameter_variation(stds, "std", "std", redraw_conf)
    parameter_variation(dimensions,
                        "dimension", "dimension", redraw_conf)
    parameter_variation(n_components, "n_components",
                        "n_components", redraw_conf)
