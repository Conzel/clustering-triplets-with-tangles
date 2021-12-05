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
import matplotlib.pyplot as plt
import numpy as np
import src.data_types as data_types
import src.utils as utils
import src.tree_tangles as tree_tangles
import src.cost_functions as cost_functions
import src.plotting as plotting
import sklearn
import yaml
import plotly.graph_objects as go
from functools import partial
from questionnaire import generate_questionnaire, ImputationMethod
from data_generation import generate_gmm_data_fixed_means, generate_gmm_data_draw_means

import sklearn.metrics
plt.style.use('ggplot')

class Configuration():
    def __init__(self, n, n_runs, seed, means, std, agreement,
                 name, num_distance_function_samples, noise, density,
                 redraw_means, min_cluster_dist, dimension,
                 n_components,
                 base_folder="results",
                 imputation_method="random"):
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
        self.imputation_method = imputation_method

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
    if conf.redraw_means:
        data = generate_gmm_data_draw_means(
            n = conf.n, std = conf.std, seed = conf.seed, 
            dimension = conf.dimension, min_cluster_dist = conf.min_cluster_dist, 
            components = conf.n_components)
    else:
        data = generate_gmm_data_fixed_means(
            n = conf.n, means = np.array(conf.means), std = conf.std, seed = conf.seed)

    # Creating the questionnaire from the data
    questionnaire = generate_questionnaire(
        data, noise=conf.noise, density=conf.density, seed=conf.seed, imputation_method=ImputationMethod(conf.imputation_method)).values

    # Interpreting the questionnaires as cuts and computing their costs
    bipartitions = data_types.Cuts((questionnaire == 1).T)
    cuts = utils.compute_cost_and_order_cuts(bipartitions, partial(
        cost_functions.mean_manhattan_distance, questionnaire, conf.num_distance_function_samples))

    # Building the tree, contracting and calculating predictions
    tangles_tree = tree_tangles.tangle_computation(cuts=cuts,
                                                   agreement=conf.agreement,
                                                   verbose=0  # print everything
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

def parameter_variation(parameter_values, name, attribute_name, base_config, plot=True, logx=False):
    """
    Runs multiple experiments varying the given parameter. The results depicted in a 
    plot (with x = parameter values, y = nmi/ars). They are also saved in a csv file.
    The output is found under the base_config name + parameter_variation

    parameter_values: list of values the parameter can take
    name: name of the parameter (is used to save the folder and to label the plot)
    attribute_name: name of the attribute of the configuration to change
    base_config: Configuration object where we vary the parameter
    plot: Set to true if plots should be saved
    logx: Determines if the parameter value should have a logarithmic scale in the plot.
    """
    ars_values = []
    nmi_values = []
    seed = base_config.seed
    base_folder = os.path.join("results", f"{base_config.name}-{name}_variation")

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
    if plot:
        # Plotting with matplotlib 
        plt.figure()
        plt.plot(parameter_values, ars_values, "--^", label="ARS")
        plt.plot(parameter_values, nmi_values, "--o", label="NMI")
        if logx:
            plt.xscale("log")
        plt.title(f"{name} variation")
        plt.legend()
        plt.savefig(os.path.join(base_folder, f"{name}_variation.png"))

        # alternative with plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=parameter_values, y=ars_values,mode="lines+markers", name="ARS"))
        fig.add_trace(go.Scatter(x=parameter_values, y=nmi_values,mode="lines+markers", name="NMI"))
        fig.update_layout(title=f"{name} variation",
            xaxis_title=f"{name}",
            yaxis_title="NMI/ARS")
        if logx:
            fig.update_xaxes(type="log")
        fig.write_html(os.path.join(base_folder, f"{name}_variation.html"))

    return ars_values, nmi_values