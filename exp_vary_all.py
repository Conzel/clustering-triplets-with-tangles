#!/usr/bin/env python3
"""
Varies all different sensible parameters of the data generation with a predefined range when run.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np
import sys
plt.style.use('ggplot')

from experiment_runner import Configuration, parameter_variation

if __name__ == "__main__":
    base_config = Configuration.from_yaml(
        yaml.load(open("experiments/06-base-config.yaml")))
    base_config.redraw_means = True
    if len(sys.argv) > 1 and (sys.argv[1] == "-p" or sys.argv[1] == "--parallelize"):
        workers = None
    else:
        workers = 1

    # Varying the agreement parameter
    agreements = list(range(1, 40, 1))
    noise = np.arange(0, 1, 0.05)
    density = np.logspace(-3, 0, num=20)
    minimum_cluster_distances = np.arange(0.5, 5, 0.5)
    stds = np.arange(0.1, 5.1, 0.5)
    dimensions = np.arange(1, 11, 1)
    n_components = np.arange(2, 11, 1)

    parameter_variation(agreements, "agreement", "agreement",
                        base_config, workers=workers)
    parameter_variation(noise, "noise", "noise", base_config)
    parameter_variation(density, "density",
                        "density", base_config, logx=True, workers=workers)
    parameter_variation(minimum_cluster_distances,
                        "min_cluster_dist", "min_cluster_dist", base_config, workers=workers)
    parameter_variation(noise, "noise", "noise", base_config, workers=workers)
    parameter_variation(stds, "std", "std", base_config, workers=workers)
    parameter_variation(dimensions,
                        "dimension", "dimension", base_config, workers=workers)
    parameter_variation(n_components, "n_components",
                        "n_components", base_config, workers=workers)
