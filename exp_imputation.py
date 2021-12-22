#!/usr/bin/env python3
"""
Experiment on imputation methods.
"""
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yaml

from experiment_runner import Configuration, parameter_variation

plt.style.use("ggplot")

if __name__ == '__main__':
    base_config = Configuration.from_yaml(
        yaml.load(open("experiments/08-imputation-base.yaml")))
    os.makedirs("results/08-imputation-results", exist_ok=True)

    if len(sys.argv) > 1 and (sys.argv[1] == "-p" or sys.argv[1] == "--parallelize"):
        workers = None
    else:
        workers = 1

    noise_values = [0.0, 0.001, 0.005, 0.01, 0.02,
                    0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    plt.figure()
    fig = go.Figure()
    imputation_methods = ["mean", "random"]
    for k in [1, 3, 5, 7]:
        imputation_methods.append(str(k) + "-NN")

    for m in imputation_methods:
        imputation_config = deepcopy(base_config)
        imputation_config.imputation_method = m

        variation_result = parameter_variation(
            noise_values, "noise", "noise", imputation_config, plot=False, workers=workers)
        ars = variation_result.ars_means
        nmi = variation_result.nmi_means

        # Plotly
        fig.add_trace(go.Scatter(x=noise_values, y=ars,
                      mode="lines+markers", name="ARS " + m))
        fig.add_trace(go.Scatter(x=noise_values, y=nmi,
                      mode="lines+markers", name="NMI " + m))
        # Matplotlib
        plt.plot(noise_values, nmi, "--o", label=("NMI " +
                 m))
        plt.plot(noise_values, ars, "--^", label=("ARS " + m))

    # Plotting one baseline, shouldn't matter too much which one we pick
    if variation_result.has_baseline():
        fig.add_trace(go.Scatter(x=noise_values, y=variation_result.nmi_means_baseline,
                                 mode="lines+markers", name="NMI baseline"))
        fig.add_trace(go.Scatter(x=noise_values, y=variation_result.ars_means_baseline,
                                 mode="lines+markers", name="ARS baseline"))
        plt.plot(noise_values, variation_result.nmi_means_baseline,
                 "--o", label="NMI baseline")
        plt.plot(noise_values, variation_result.ars_means_baseline,
                 "--^", label="ARS baseline")

    fig.update_layout(title=f"Comparison of imputation methods",
                      xaxis_title=f"Noise",
                      yaxis_title="NMI/ARS")
    fig.write_html("./results/08-imputation-results/imputations.html")
    plt.legend()
    plt.savefig("results/08-imputation-results/imputations.png")
