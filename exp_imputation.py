#!/usr/bin/env python3
"""
Experiment on imputation methods.
"""
from experiment_runner import Configuration, parameter_variation
import yaml
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from copy import deepcopy
plt.style.use("ggplot")

if __name__ == '__main__':
    base_config = Configuration.from_yaml(
        yaml.load(open("experiments/08-imputation-base.yaml")))
    os.makedirs("results/08-imputation-results", exist_ok=True)

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
            noise_values, "noise", "noise", imputation_config, plot=False)
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

    fig.update_layout(title=f"Comparison of imputation methods",
                      xaxis_title=f"Noise",
                      yaxis_title="NMI/ARS")
    plt.legend()
    fig.write_html("./results/08-imputation-results/imputations.html")
    plt.savefig("results/08-imputation-results/imputations.png")
