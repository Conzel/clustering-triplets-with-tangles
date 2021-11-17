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
plt.style.use('ggplot')

from data_generation import Configuration, run_experiment

base_config = Configuration.from_yaml(
    yaml.load(open("experiments/06-base-config.yaml")))


def parameter_variation(parameter_values, name, attribute_name, n_runs=1):
    ars_values = []
    nmi_values = []
    seed = base_config.seed
    base_folder = os.path.join("results", f"06-{name}_variation")


    for p in parameter_values:
        print(f"Calculating for {name} variation, value: {p}")
        ars_sum = 0
        nmi_sum = 0
        for i in range(n_runs):
            conf = copy.deepcopy(base_config)
            setattr(conf, attribute_name, p)
            conf.name = f"{name}-{p}-{i}"
            conf.base_folder = base_folder
            conf.seed = seed + i

            ars, nmi = run_experiment(conf)
            ars_sum += ars
            nmi_sum += nmi

        ars_values.append(ars_sum / n_runs)
        nmi_values.append(nmi_sum / n_runs)

    metric_results = {f"{name}": parameter_values, 'nmi': nmi_values, 'ars': ars_values}
    df = pd.DataFrame(data=metric_results)
    df.to_csv(os.path.join(base_folder, "metric_results.txt"))

    plt.figure()
    plt.plot(parameter_values, ars_values, "--^", label="ARS")
    plt.plot(parameter_values, nmi_values, "--o", label="NMI")
    plt.title(f"{name} variation")
    plt.legend()
    plt.savefig(os.path.join(base_folder, f"{name}_variation.png"))


if __name__ == "__main__":
    # Varying the agreement parameter
    agreements = list(range(1, 21, 2))
    noise = np.arange(0, 1, 0.05)
    density = np.logspace(-3, 0, num=20)

    # parameter_variation(agreements, "agreement", "agreement", n_runs=10)
    # parameter_variation(noise, "noise", "noise", n_runs=10)
    parameter_variation(density, "density", "density", n_runs=10)
