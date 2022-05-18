"""
This module contains utility functions and classes that help in creating
the plots shown in the thesis.
"""
from __future__ import annotations
import argparse
import os
from datasets import Dataset
from typing import Optional
from pathlib import Path
from comparison_hc import ComparisonHC
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from cblearn.embedding import SOE, CKL, GNMDS, FORTE, TSTE, MLDS
from questionnaire import Questionnaire
from estimators import LandmarkTangles, MajorityTangles, OrdinalTangles
from triplets import subsample_triplets, triplets_to_majority_neighbour_cuts, unify_triplet_order
import pandas as pd


SEED = 1
RUNS_AVERAGED = 3


class DataCache:
    """
    Simple cache structure to not reload our data if unnecessary.
    """

    def __init__(self, results_folder: str, exp_name: str, verbose: bool = True):
        self.exp_name = exp_name
        self.data: Optional[pd.DataFrame] = None
        self.results_folder = Path(results_folder)
        self.verbose = verbose

    def save(self, data: pd.DataFrame):
        """
        Caches the data produced data.
        """
        self.data = data
        data.to_csv(self.results_folder / f"{self.exp_name}.csv")

    def load(self) -> bool:
        """
        Loads the data from the cache.
        """
        if self.data is not None:
            return True
        elif not self.results_folder.exists():
            return False
        elif self.exp_name + ".csv" not in os.listdir(self.results_folder):
            return False
        else:
            if self.verbose:
                print(f"Previous experiment result found.")
                print(
                    f"Loading data from cache at {self.results_folder / self.exp_name}.csv...")
            self.data = pd.read_csv(
                self.results_folder / f"{self.exp_name}.csv")
            return True


class ClusteringEvaluationSuite:
    """
    Evaluates the performance of different clustering algorithms and reports their
    performance in a pandas dataframe.

    The clustering algorithms used are OrdinalTangles, LandmarkTangles, as well as
    SOE, CKL, GNMDS, FORTE, TSTE, MLDS in conjunction with a given clusterer (f.e. KMeans).
    """

    def __init__(self, agreement: int, embedding_dim: int, clusterer: sklearn.base.ClusterMixin, seed: int, radius: float = 1 / 2, methods_to_include: Optional[list[str]] = None, methods_to_exclude: Optional[list[str]] = None, imputation: Optional[str] = None):
        """
        Args:
            agreement: Agreement parameter of both tangles methods
            embedding_dim: Dimension of the embedding in the Ordinal embedding methods
            n_clusters: Number of clusters in the kMeans clustering method
        """
        if methods_to_include is not None and methods_to_exclude is not None:
            raise ValueError("White and blacklist both specified. Choose one.")
        self.embedding_dim = embedding_dim
        self.clusterer = clusterer
        self.seed = seed
        self.agreement = agreement
        self.radius = radius
        self.imputation = imputation
        self.evaluators: list[sklearn.base.ClusterMixin] = []
        self.names: list[str] = []
        self._add_evaluators(methods_to_include)

    def _add_evaluators(self, methods_to_include: Optional[list[str]] = None, methods_to_exclude: Optional[list[str]] = None):
        """
        Adds all evaluators to the suite that are in names_to_evaluate. If None, all
        evaluators are set.
        """
        all_names = ["L-Tangles", "M-Tangles", "ComparisonHC",
                     "SOE", "CKL", "GNMDS", "FORTE", "TSTE", "MLDS"]
        embedders = [SOE(n_components=self.embedding_dim, random_state=self.seed), CKL(n_components=self.embedding_dim, random_state=self.seed),
                     GNMDS(n_components=self.embedding_dim, random_state=self.seed), FORTE(
            n_components=self.embedding_dim, random_state=self.seed), TSTE(n_components=self.embedding_dim, random_state=self.seed),
            MLDS(n_components=1, random_state=self.seed)]
        all_evaluators = [LandmarkTangles(agreement=self.agreement, imputation=self.imputation), MajorityTangles(
            agreement=self.agreement, radius=self.radius), ComparisonHC(num_clusters=self.clusterer.get_params()["n_clusters"])]
        for embedder in embedders:
            all_evaluators.append(Pipeline([("embedder", embedder), ("clusterer",
                                                                     self.clusterer)]))

        if methods_to_include is None and methods_to_exclude is None:
            self.evaluators = all_evaluators
            self.names = all_names
        else:
            self.evaluators = []
            self.names = []
            for i, name in enumerate(all_names):
                if (methods_to_include is not None and name in methods_to_include) or (methods_to_exclude is not None and name not in methods_to_exclude):
                    self.evaluators.append(all_evaluators[i])
                    self.names.append(name)

    def score_all_once(self, triplets: np.ndarray, responses: np.ndarray, target: np.ndarray) -> pd.DataFrame:
        """
        Returns a dataframe containing the results of all embedders applied
        to the given triplets.

        The dataframe is in wide format, and each row contains:
        method: str, name of the method used.
        nmi: float, normalized mutual information.
        ars: float, adjusted rand score.
        """
        rows = []
        for name, evaluator in zip(self.names, self.evaluators):
            pred = evaluator.fit_predict(triplets, responses)

            nmi = normalized_mutual_info_score(pred, target)
            ars = adjusted_rand_score(pred, target)
            rows.append(dict(method=name, nmi=nmi, ars=ars))
        return pd.DataFrame(rows)

    def score_all(self, data_generator) -> pd.DataFrame:
        """
        Runs all entries from the data generator and returns a dataframe that
        contains the results of all runs.

        Args:
            data_generator: an iterable that yields items of the form
                (triplets, responses, target, {run_denoms}). Must be finite, else
                the function does not stop.
                Run denoms is a dictionary that contains the denominators for each run
                (run number, density, noise, ...)

        Returns: 
            Dataframe with columns as described in score_all_once, 
            with the run number added.
        """
        run = 0
        dfs = []
        for triplets, responses, target, run_denoms in data_generator:
            dfs.append(self.score_all_once(
                triplets, responses, target).assign(**run_denoms))
            run += 1
        return pd.DataFrame(pd.concat(dfs, axis=0))


def simulation_all_triplets_gauss(debug: bool, n_runs=RUNS_AVERAGED):
    """
    Simulates using all triplets on a gaussian dataset. As this is
    a very simple experiment, no line plot is produced, just an assignment
    of the final clustering.
    """
    data = Dataset.get(Dataset.GAUSS_SMALL)
    q = Questionnaire.from_metric(data.xs)
    triplets, responses = q.to_bool_array(return_responses=True)
    df = evaluation_embedders(triplets, responses,
                              embedding_dim=2, n_clusters=3, target=data.ys, seed=SEED)
    tangles = OrdinalTangles(agreement=8)
    ys_tangles = tangles.fit_predict(q.values)
    # saving raw data
    df = pd.concat((df, tangles_result_to_row(
        ys_tangles, data.ys, TanglesMethod.DIRECT)), ignore_index=True).assign(experiment="sim-all-gauss")
    df.to_csv(RESULTS_FOLDER / "sim_all_gauss.csv")
    # saving plot
    plot_assignments(data.xs, data.ys)
    save_plot("gauss_small")
    return df


def _sim_lower_density(dataset: int, agreement: int, densities: list[float], n_runs: int) -> pd.DataFrame:
    """
    Helper function for simulating lowering the density. Different datasets can be plugged in.
    """
    dfs = []
    j = 0
    for density in densities:
        print(f"Density = {density}")
        for _ in range(n_runs):
            data = Dataset.get(dataset, SEED + j)
            j += 1
            q = Questionnaire.from_metric(
                data.xs, density=density, verbose=False)

            triplets, responses = q.to_bool_array(return_responses=True)
            df = evaluation_embedders(triplets, responses,
                                      embedding_dim=2, n_clusters=3, target=data.ys, seed=SEED + j).assign(density=density)
            dfs.append(df)

            tangles = OrdinalTangles(agreement=agreement)
            ys_tangles = tangles.fit_predict(q.values)
            dfs.append(tangles_result_to_row(
                ys_tangles, data.ys, TanglesMethod.DIRECT).assign(density=density))

            dfs.append(make_soe_random(data, q.values.size,
                       SEED + j).assign(density=density))
    return pd.concat(dfs, ignore_index=True)


def simulation_lowering_density_gauss_small(debug: bool, n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    """
    Simulates lowering the density on a small gaussian dataset.
    Lowering density means that we take out progressively more columns from the final questionnaire.
    """
    if debug:
        densities = [0.5, 0.1, 0.01]
    else:
        densities = np.logspace(0, -4, 20).tolist()
    df = _sim_lower_density(Dataset.GAUSS_SMALL, 8, densities, n_runs).assign(
        experiment="sim-lower-density-gauss-small")
    df.to_csv(RESULTS_FOLDER / "sim-lower-density-gauss-small.csv")
    plot_line(df, "density", "nmi")
    save_plot("sim-lower-density-gauss-small")
    return df


def simulation_lowering_density_gauss_large(debug: bool, n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    """
    Analog to the version with the small dataset.
    """
    if debug:
        densities = [0.0001, 0.00008, 0.00005]
    else:
        densities = np.logspace(-3, -5, 10).tolist()
    df = _sim_lower_density(Dataset.GAUSS_LARGE, 80, densities, n_runs).assign(
        experiment="sim-lower-density-gauss-large")
    df.to_csv(RESULTS_FOLDER / "sim-lower-density-gauss-large.csv")
    plot_line(df, "density", "nmi")
    save_plot("sim-lower-density-gauss-large")
    return df


def simulation_adding_noise_gauss(debug: bool, n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    """
    Simulates adding noise on a small gaussian dataset.
    """
    dfs = []
    j = 0
    if debug:
        noises = [0.05, 0.1, 0.15, 0.2]
    else:
        noises = np.arange(0, 0.51, 0.01)
    for noise in noises:
        print(f"Noise = {noise}")
        for _ in range(n_runs):
            data = Dataset.get(Dataset.GAUSS_SMALL, SEED + j)
            j += 1
            q = Questionnaire.from_metric(
                data.xs, verbose=False, noise=noise, flip_noise=True)
            triplets, responses = q.to_bool_array(return_responses=True)
            df = evaluation_embedders(
                triplets, responses, embedding_dim=2, n_clusters=3, target=data.ys, seed=SEED + j).assign(noise=noise)
            dfs.append(df)

            tangles = OrdinalTangles(agreement=8)
            ys_tangles = tangles.fit_predict(q.values)
            dfs.append(tangles_result_to_row(
                ys_tangles, data.ys, TanglesMethod.DIRECT).assign(noise=noise))

    df = pd.concat(dfs, ignore_index=True).assign(
        experiment="sim-adding-noise-gauss")
    df.to_csv(RESULTS_FOLDER / "sim-adding-noise-gauss.csv")
    plot_line(df, "noise", "nmi")
    save_plot("sim-adding-noise-gauss")
    return df


def simulation_majority_cuts(debug: bool, n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    """
    Outputs how the majority cuts would look like and their clustering
    against number of triplets.
    """
    if debug:
        num_triplets = [100, 500, 1000]
    else:
        num_triplets = [100, 500, 1000, 5000, 10000, 20000, 50000]
    dfs = []

    j = 0
    for n in num_triplets:
        print(f"#triplets = {n}")
        for k in range(n_runs):
            data = Dataset.get(Dataset.GAUSS_SMALL, SEED + j)
            j += 1
            triplets, responses = subsample_triplets(data.xs, n)
            unified_triplets = unify_triplet_order(triplets, responses)
            cuts = triplets_to_majority_neighbour_cuts(
                unified_triplets, radius=1 / 2)
            df = evaluation_embedders(
                triplets, responses, embedding_dim=2, n_clusters=3, target=data.ys, seed=SEED + j).assign(**{r"\# triplets": n})
            dfs.append(df)

            tangles = OrdinalTangles(agreement=8)
            ys_tangles = tangles.fit_predict(cuts)
            dfs.append(tangles_result_to_row(
                ys_tangles, data.ys, TanglesMethod.MAJORITY_CUT).assign(**{r"\# triplets": n}))

    df = pd.concat(dfs, ignore_index=True).assign(
        experiment="sim-majority-cuts")
    df.to_csv(RESULTS_FOLDER / "sim-majority-cuts.csv")
    plot_line(df, r"\# triplets", "nmi")
    save_plot("sim-majority-cuts")
    return df


def simulation_majority_cuts_appearance(debug: bool, n_runs=RUNS_AVERAGED):
    num_triplets = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000]

    data = Dataset.get(Dataset.GAUSS_SMALL, SEED)
    for n in num_triplets:
        print(f"#triplets = {n}")
        triplets, responses = subsample_triplets(data.xs, n)
        unified_triplets = unify_triplet_order(triplets, responses)
        cuts = triplets_to_majority_neighbour_cuts(
            unified_triplets, radius=1 / 2)

        p = 8
        plot_assignments(data.xs, cuts[p, :])
        plt.plot(data.xs[p, 0], data.xs[p, 1], "kx", markersize=14)
        save_plot(f"majority-cut-{p}-n_triplets-{n}")


def simulation_majority_cuts_adding_noise(debug: bool, n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    dfs = []
    j = 0
    n = 20000  # majority triplets has reasonable performance there
    if debug:
        noises = [0.05, 0.1, 0.15, 0.2]
    else:
        noises = np.arange(0, 0.51, 0.01)
    for noise in noises:
        print(f"Noise = {noise}")
        for _ in range(n_runs):
            data = Dataset.get(Dataset.GAUSS_SMALL, SEED + j)
            j += 1
            triplets, responses = subsample_triplets(data.xs, n, noise=noise)
            unified_triplets = unify_triplet_order(triplets, responses)
            cuts = triplets_to_majority_neighbour_cuts(
                unified_triplets, radius=1 / 2)
            df = evaluation_embedders(
                triplets, responses, embedding_dim=2, n_clusters=3, target=data.ys, seed=SEED + j).assign(noise=noise)
            dfs.append(df)

            tangles = OrdinalTangles(agreement=8)
            ys_tangles = tangles.fit_predict(cuts)
            dfs.append(tangles_result_to_row(
                ys_tangles, data.ys, TanglesMethod.MAJORITY_CUT).assign(noise=noise))
    df = pd.concat(dfs, ignore_index=True).assign(
        experiment="sim-adding-noise-gauss")
    df.to_csv(RESULTS_FOLDER / "sim-majority-noise.csv")
    plot_line(df, "noise", "nmi")
    save_plot("sim-majority-noise")
    return df


def simulation_adding_noise_and_lowering_density(debug: bool, n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    """
    Simulates on the small gaussian dataset:
    Lower density and simultaneously adds noise.
    """
    dfs = []
    if debug:
        noises = [0.05, 0.1]
        densities = [0.01, 0.1]
    else:
        noises = np.arange(0, 0.41, 0.02)
        densities = np.logspace(0, -4, 20)
    j = 0
    for noise in noises:
        for density in densities:
            print(f"Density = {density}, Noise = {noise}")
            for _ in range(n_runs):
                data = Dataset.get(Dataset.GAUSS_SMALL, SEED + j)
                j += 1
                q = Questionnaire.from_metric(
                    data.xs, density=density, verbose=False, noise=noise, flip_noise=True)
                triplets, responses = q.to_bool_array(return_responses=True)
                df = evaluation_embedders(
                    triplets, responses, embedding_dim=2, n_clusters=3, target=data.ys, seed=SEED + j, names_to_evaluate=["SOE"]).assign(noise=noise, density=density)
                dfs.append(df)

                tangles = OrdinalTangles(agreement=8)
                ys_tangles = tangles.fit_predict(q.values)
                dfs.append(tangles_result_to_row(
                    ys_tangles, data.ys, TanglesMethod.DIRECT).assign(noise=noise, density=density))
                dfs.append(make_soe_random(data, q.values.size,
                                           SEED + j).assign(density=density, noise=noise))
    df = pd.concat(dfs, ignore_index=True).assign(
        experiment="sim-adding-noise-lowering-density-gauss")
    df.to_csv(RESULTS_FOLDER / "sim-noise-density.csv")
    for method in ["TANGLES-DIRECT", "SOE", "SOE-RANDOM"]:
        plot_heatmap(df, "density", "noise", "nmi", method)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xscale("log")
        save_plot(f"sim-noise-density-{method.lower()}")
    return df


def all_experiments(debug: bool, n_runs=RUNS_AVERAGED):
    dfs = []
    for exp in available_experiments:
        if exp == "all":
            continue
        print(f"\nRUNNING: {exp}\n{50 * '-'}\n")
        dfs.append(experiment_function_binding[exp](
            debug=debug, n_runs=n_runs))
    return pd.concat(dfs)


available_experiments = ["sim-all-gauss", "sim-lower-density-small",
                         "sim-lower-density-large", "sim-noise", "sim-noise-density",
                         "sim-majority", "sim-majority-cuts-appearance", "sim-majority-noise",
                         "all"]

experiment_function_binding = {
    "sim-all-gauss": simulation_all_triplets_gauss,
    "sim-lower-density-small": simulation_lowering_density_gauss_small,
    "sim-lower-density-large": simulation_lowering_density_gauss_large,
    "sim-noise": simulation_adding_noise_gauss,
    "sim-noise-density": simulation_adding_noise_and_lowering_density,
    "sim-majority": simulation_majority_cuts,
    "sim-majority-cuts-appearance": simulation_majority_cuts_appearance,
    "sim-majority-noise": simulation_majority_cuts_adding_noise,
    "all": all_experiments,
}


def main(experiment: list[str], debug: bool, n_runs: int):
    # validation for quick-fail
    for exp in experiment:
        if exp not in experiment_function_binding.keys():
            raise ValueError(f"Experiment {experiment} not implemented yet.")
    # all good, can proceed
    for exp in experiment:
        if debug:
            experiment_function_binding[exp](n_runs=1, debug=True)
        else:
            experiment_function_binding[exp](n_runs=n_runs, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the thesis experiments")
    parser.add_argument("experiment", nargs="+",
                        choices=available_experiments, help="Experiment to run")
    parser.add_argument(
        "--debug", action="store_true", help="Activates debug runs. Runs every experiment only once and with a reduced parameter set, if applicable. This is designed to make the experiments run fast as a priority.")
    parser.add_argument("--runs", type=int, default=RUNS_AVERAGED)
    args = parser.parse_args()
    main(args.experiment, args.debug, args.runs)
