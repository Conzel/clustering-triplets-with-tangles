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
    for n in noise:
        print(n)
        base_config.noise = n
        variation = parameter_variation(agreements, "agreement", "agreement",
                                        base_config, workers=None)
        df = variation.to_df()
        df["noise"] = n
        variations.append(df)

    result = pd.concat(variations)
    alt.Chart(result).mark_line(
        point={
            "filled": False,
            "fill": "white"
        }
    ).encode(x="noise", y="nmi", color="agreement:N").show()
