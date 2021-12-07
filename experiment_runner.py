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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
import yaml
import plotly.graph_objects as go
from functools import partial
from questionnaire import generate_questionnaire, ImputationMethod
from data_generation import generate_gmm_data_fixed_means, generate_gmm_data_draw_means
from multiprocessing import Pool

import sklearn.metrics
plt.style.use('ggplot')


class Configuration():
    def __init__(self, n, n_runs, seed, means, std, agreement,
                 name, num_distance_function_samples, noise, density,
                 redraw_means, min_cluster_dist, dimension,
                 n_components,
                 base_folder="results",
                 imputation_method="random",
                 baseline="none"):
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
        self.baseline = baseline

    def from_yaml(yaml_dict):
        return Configuration(**yaml_dict)

    def __str__(self) -> str:
        return "Configuration: " + str(self.__dict__)


class HardClusteringEvaluation():
    def __init__(self, y, y_pred) -> None:
        self.nmi = normalized_mutual_info_score(y, y_pred)
        self.ars = adjusted_rand_score(y, y_pred)


class Baseline():
    def __init__(self, name):
        self.name = name
        if name.lower() == "none" or name is None:
            self.method = Baseline._raise_no_baseline_error
        if name.lower() == "gmm":
            self.method = Baseline._gmm_baseline

    def _raise_no_baseline_error(data: data_types.Data, n_components):
        raise ValueError("No baseline")

    def _gmm_baseline(data: data_types.Data, n_components, seed=None):
        """
        Calculates a baseline for the clustering by fitting a gaussian mixture model
        to the data x and inferring labels y'. 

        Returns ARS and NMI of the GMM, using the given ground truth labels y to calculate the
        metrics. 
        """
        gm = GaussianMixture(n_components=n_components,
                             random_state=seed).fit(data.xs)
        y_pred = gm.predict(data.xs)
        return y_pred

    def predict(self, data: data_types.Data, n_components) -> np.ndarray:
        """
        Uses the chosen baseline to predict the labels of the given data.
        """
        return self.method(data, n_components)


class ExperimentResult():
    """
    Result of an experiment ran multiple times. Properties:
    ars_values: list of the ARS values of every single run
    nmi_values: list of the NMI values of every single run
    ars_mean: mean of the ARS values
    nmi_mean: mean of the NMI values
    ars_std: standard deviation of the ARS values
    nmi_std: standard deviation of the NMI values
    """

    def __init__(self, run_results: list):
        # normal results
        self.ars_values = [t.ars for t in run_results]
        self.nmi_values = [t.nmi for t in run_results]
        assert len(self.ars_values) == len(self.nmi_values)
        ars_np = np.array(self.ars_values)
        nmi_np = np.array(self.nmi_values)
        self.ars_mean = np.mean(ars_np)
        self.nmi_mean = np.mean(nmi_np)
        self.ars_std = np.std(ars_np)
        self.nmi_std = np.std(nmi_np)

        self._all_results_have_baseline = all([r.has_baseline for r in run_results])

        if self._all_results_have_baseline:
            # baseline
            self.ars_values_baseline = [t.ars_baseline for t in run_results]
            self.nmi_values_baseline = [t.nmi_baseline for t in run_results]
            assert len(self.ars_values_baseline) == len(self.nmi_values_baseline)
            ars_np_baseline = np.array(self.ars_values_baseline)
            nmi_np_baseline = np.array(self.nmi_values_baseline)
            self.ars_mean_baseline = np.mean(ars_np_baseline)
            self.nmi_mean_baseline = np.mean(nmi_np_baseline)
            self.ars_std_baseline = np.std(ars_np_baseline)
            self.nmi_std_baseline = np.std(nmi_np_baseline)

    def has_baseline(self):
        return self._all_results_have_baseline

    def to_df(self):
        """
        Returns a pandas dataframe of the data (only raw data, no aggregates such as
        mean, std).

        The resulting dataframe contains the nmi and ars values for every run.
        """
        df = pd.DataFrame({"run": list(range(len(self.ars_values))),
                           "ars": self.nmi_values, "nmi": self.nmi_values,
                           "ars_baseline": self.ars_values_baseline,
                           "nmi_baseline": self.nmi_values_baseline})
        return df

    def save_csv(self, filepath):
        """
        Saves experiment results to a csv file. 
        """
        self.to_df().to_csv(os.path.join(filepath), index=False)


class RunResult():
    """
    Result of a single run of an experiment. Properties:
    ars: adjusted rand score
    nmi: normalized mutual information
    """

    def __init__(self, run_evaluation: HardClusteringEvaluation, baseline_evaluation: HardClusteringEvaluation = None):
        self.ars = run_evaluation.ars
        self.nmi = run_evaluation.nmi
        self._has_baseline = baseline_evaluation is not None
        if baseline_evaluation is not None:
            self.ars_baseline = baseline_evaluation.ars
            self.nmi_baseline = baseline_evaluation.nmi
    def has_baseline(self) -> bool:
        return self._has_baseline

class VariationResults():
    """
    Represents the results of the parameter variation.
    """

    def __init__(self, parameter_name: str, parameter_values: list, experiment_results: "list(ExperimentResult)"):
        self.parameter_values = parameter_values
        self.parameter_name = parameter_name
        self._experiment_results = experiment_results
        # Getting the singular results out of the experiments
        self.nmi_means = [r.nmi_mean for r in experiment_results]
        self.ars_means = [r.ars_mean for r in experiment_results]
        self.ars_stds = [r.ars_std for r in experiment_results]
        self.nmi_stds = [r.nmi_std for r in experiment_results]

        self._all_results_have_baseline = all([r.has_baseline() for r in experiment_results])
        if self._all_results_have_baseline:
            self.nmi_means_baseline = [r.nmi_mean_baseline for r in experiment_results]
            self.ars_means_baseline = [r.ars_mean_baseline for r in experiment_results]
            self.ars_stds_baseline = [r.ars_std_baseline for r in experiment_results]
            self.nmi_stds_baseline = [r.nmi_std_baseline for r in experiment_results]


    def has_baseline(self) -> bool:
        return self._all_results_have_baseline

    def plot(self, base_folder, logx=False):
        """
        Plots the results. Results are saved under the given base folder
        """
        # Plotting with matplotlib
        plt.figure()
        plt.plot(self.parameter_values, self.ars_means, "--^", label="ARS")
        plt.plot(self.parameter_values, self.nmi_means, "--o", label="NMI")
        if self.has_baseline():
            plt.plot(self.parameter_values, self.ars_means_baseline, "--^", label="Baseline ARS")
            plt.plot(self.parameter_values, self.nmi_means_baseline, "--o", label="Baseline NMI")
        
        if logx:
            plt.xscale("log")
        plt.title(f"{self.parameter_name} variation")
        plt.legend()
        plt.savefig(os.path.join(
            base_folder, f"{self.parameter_name}_variation.png"))

        # alternative with plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.parameter_values,
                      y=self.ars_means, mode="lines+markers", name="ARS"))
        fig.add_trace(go.Scatter(x=self.parameter_values,
                      y=self.nmi_means, mode="lines+markers", name="NMI"))
        fig.add_trace(go.Scatter(x=self.parameter_values,
                      y=self.ars_means_baseline, mode="lines+markers", name="Baseline ARS"))
        fig.add_trace(go.Scatter(x=self.parameter_values,
                      y=self.nmi_means_baseline, mode="lines+markers", name="Baseline NMI"))
        fig.update_layout(title=f"{self.parameter_name} variation",
                          xaxis_title=self.parameter_name,
                          yaxis_title="Mean NMI/ARS")
        fig.write_image(os.path.join(base_folder, f"{self.parameter_name}_variation.png"))
        fig.update_layout(title=f"{self.parameter_name} variation",
                          xaxis_title=f"{self.parameter_name}",
                          yaxis_title="NMI/ARS")
        if logx:
            fig.update_xaxes(type="log")
        print("Saving plot to", os.path.join(base_folder, f"{self.parameter_name}_variation.html"))
        fig.write_html(os.path.join(base_folder, f"{self.parameter_name}_variation.html"))


    def to_df(self):
        """
        Returns a pandas dataframe of the data (only raw data, no aggregates such as
        mean, std).

        The resulting dataframe contains the nmi and ars values for every run.
        """
        # Saving the results
        metric_results = {f"{self.parameter_name}": self.parameter_values,
                          'nmi': self.nmi_means, 'ars': self.ars_means}
        df = pd.DataFrame(data=metric_results)
        return df

    def save_csv(self, filepath):
        """
        Saves experiment results to a csv file. 
        """
        self.to_df().to_csv(os.path.join(filepath), index=False)


def run_experiment(conf: Configuration, workers=1) -> ExperimentResult:
    """
    Runs an experiment with the given configuration. 

    In this example, we generate a synthetic dataset consisting of 
    n d-dimensional Gaussians. We then create a triplet questionnaire 
    of the datapoints and use this to create bipartitions.

    We then create a hard clustering using tangles.
    We plot the clustering and evaluate the clustering quality using NMI and ARS.

    By passing in a workers argument, we can do multiple runs of the experiment in parallel,
    greatly speeding up the process (on modern machines > 2x).
    If you want to use multiple processing, we advise passing in an argument of None 
    (which will use all available cores). Multiprocessing messes 
    up the console outputs, so you sadly won't see any progress on the single experiments.

    Returns a tuple (ARS, NMI) of the resulting hard clustering.
    """
    seed = conf.seed

    if workers == 1:
        runner = _run_once_verbose
    else:
        runner = _run_once_quiet

    configs = []

    # Generating the configurations with the different seeds
    for i in range(conf.n_runs):
        backup_conf = copy.deepcopy(conf)
        backup_conf.seed = seed + i
        configs.append(backup_conf)

    # For multiprocessing
    if workers is None or workers > 1:
        print("Running parallel experiments...")
        with Pool(workers) as pool:
            run_results = list(
                tqdm(pool.imap(runner, configs), total=len(configs)))
    else:
        run_results = list(map(runner, configs))

    experiment_result = ExperimentResult(run_results)
    experiment_result.save_csv(os.path.join(
        conf.base_folder, conf.name, conf.name + "_metrics.csv"))
    return experiment_result

# These two functions are defined because pool can only pickle top level functions (not lambdas)

def _run_once_verbose(conf: Configuration):
    return _run_once(conf)


def _run_once_quiet(conf: Configuration):
    return _run_once(conf, verbose=False)


def _tangles_hard_predict(conf: Configuration, data: data_types.Data, verbose=True):
    """
    Uses the tangles algorithm to produce a hard clustering on the given data.

    Returns predicted y labels for the data points.
    """
    # Creating the questionnaire from the data
    questionnaire = generate_questionnaire(
        data, noise=conf.noise, density=conf.density, seed=conf.seed, imputation_method=ImputationMethod(
            conf.imputation_method),
        verbose=verbose).values

    # Interpreting the questionnaires as cuts and computing their costs
    bipartitions = data_types.Cuts((questionnaire == 1).T)
    cuts = utils.compute_cost_and_order_cuts(bipartitions, partial(
        cost_functions.mean_manhattan_distance, questionnaire, conf.num_distance_function_samples),
        verbose=verbose)

    # Building the tree, contracting and calculating predictions
    tangles_tree = tree_tangles.tangle_computation(cuts=cuts,
                                                   agreement=conf.agreement,
                                                   # print nothing
                                                   verbose=int(verbose)
                                                   )

    contracted = tree_tangles.ContractedTangleTree(tangles_tree)
    contracted.prune(5, verbose=verbose)

    contracted.calculate_setP()

    # soft predictions
    weight = np.exp(-utils.normalize(cuts.costs))
    tree_tangles.compute_soft_predictions_children(
        node=contracted.root, cuts=bipartitions, weight=weight, verbose=3)

    ys_predicted, _ = utils.compute_hard_predictions(
        contracted, cuts=bipartitions, verbose=verbose)

    return ys_predicted


def _generate_data(conf: Configuration):
    """
    Generates a synthetic dataset (mixture of gaussians) with the given configuration.
    """
    if conf.redraw_means:
        data = generate_gmm_data_draw_means(
            n=conf.n, std=conf.std, seed=conf.seed,
            dimension=conf.dimension, min_cluster_dist=conf.min_cluster_dist,
            components=conf.n_components)
    else:
        data = generate_gmm_data_fixed_means(
            n=conf.n, means=np.array(conf.means), std=conf.std, seed=conf.seed)

    return data


def _run_once(conf: Configuration, verbose=True) -> RunResult:
    """Runs the experiment once with the given configuration. Ignores
       n_runs parameter.

       If verbose is set to true, prints out steps in the process such as data generation etc.

       Returns a tuple (ars, nmi) of the resulting hard clustering.
    """
    np.random.seed(conf.seed)
    data = _generate_data(conf)
    y_predicted = _tangles_hard_predict(conf, data, verbose=verbose)

    # evaluate hard predictions
    assert data.ys is not None
    evaluation = HardClusteringEvaluation(data.ys, y_predicted)

    # Writing back results
    # Creating results folder if it doesn't exist
    result_output_path = Path(os.path.join(conf.base_folder, conf.name))
    result_output_path.mkdir(parents=True, exist_ok=True)
    if conf.dimension == 2:
        # Plotting the hard clustering
        plotting.plot_hard_predictions(data=data, ys_predicted=y_predicted,
                                       path=result_output_path)

    # --- Checking if we need to calculate a baseline as well ---
    baseline_evaluation = None
    if conf.baseline is not None and conf.baseline.lower() != "none":
        if verbose:
            print("Evaluating baseline...")
        baseline = Baseline(conf.baseline)
        baseline_prediction = baseline.predict(data, conf.n_components)
        baseline_evaluation = HardClusteringEvaluation(
            data.ys, baseline_prediction)

    return RunResult(evaluation, baseline_evaluation)


def parameter_variation(parameter_values, name, attribute_name, base_config, plot=True, logx=False, 
                        workers=1):
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
    experiment_results = []
    seed = base_config.seed
    base_folder = os.path.join(
        "results", f"{base_config.name}-{name}_variation")

    for p in parameter_values:
        print(f"Calculating for {name} variation, value: {p}")
        conf = copy.deepcopy(base_config)
        if not hasattr(conf, attribute_name):
            raise ValueError(f"{attribute_name} not found in {conf}")
        setattr(conf, attribute_name, p)
        conf.name = f"{name}-{p:.4f}"
        conf.base_folder = base_folder

        experiment_results.append(run_experiment(conf, workers=workers))

    result = VariationResults(parameter_name=name, parameter_values=parameter_values,
                              experiment_results=experiment_results)
    result.save_csv(os.path.join(base_folder, "metric_results.txt"))
    if plot:
        result.plot(base_folder=base_folder, logx=logx)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("First argument has to be the name of a YAML configuration. Exiting.")
        exit(1)
    # Loading the configuration
    with open(sys.argv[1], "r") as f:
        conf = Configuration.from_yaml(yaml.safe_load(f))

    if len(sys.argv) > 2 and (sys.argv[2] == "-p" or sys.argv[2] == "--parallel"):
        workers = None
    else:
        workers = 1

    # Running the experiment
    run_experiment(conf, workers=workers)
