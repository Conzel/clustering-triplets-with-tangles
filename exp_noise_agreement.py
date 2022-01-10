#!/usr/bin/env python3
"""
Plots noise over different agreement values.
"""

import altair as alt
import pandas as pd
from experiment_runner import Configuration, parameter_variation
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.style.use('ggplot')

if __name__ == "__main__":
    base_config = Configuration.from_yaml(
        yaml.load(open("experiments/10-noise-agreement.yaml")))
    base_config.redraw_means = True
    baseline = "soe-knn"

    if len(sys.argv) > 1 and (sys.argv[1] == "-p" or sys.argv[1] == "--parallelize"):
        workers = None
    else:
        workers = 1

    # Varying the agreement parameter
    agreements = list(range(2, 10, 1))
    noise = np.arange(0.1, 0.5, 0.05)
    base_config.baseline = "none"
    base_config.n_runs = 5

    variations = []
    for a in agreements:
        if a == agreements[-1]:
            base_config.baseline = baseline
        print(a)
        base_config.agreement = a
        variation = parameter_variation(noise, "noise", "noise",
                                        base_config, workers=None)
        df = variation.to_df()
        df["agreement"] = a
        variations.append(df)

    baseline_df = df.copy()
    baseline_df["nmi_means_baseline"] = variation.nmi_means_baseline
    # for the legend
    baseline_df["agreement"] = baseline

    result = pd.concat(variations)
    variation_result = alt.Chart(result).mark_line(
        point={
            "filled": False,
            "fill": "white"
        }
    ).encode(x="noise", y="nmi", color="agreement:N")

    baseline_result = alt.Chart(baseline_df).mark_line(
        point={
            "filled": False,
            "fill": "white"
        }
    ).encode(x="noise", y="nmi_means_baseline", color="agreement:N")

    (baseline_result + variation_result).show()
