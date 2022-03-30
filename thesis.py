"""
File that is able to produce all the plots for the thesis.

Produce plot via calling the script and then calling
--exp <name>
"""
from __future__ import annotations
import argparse
from audioop import add
from typing import Optional, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from prometheus_client import Enum
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from data_generation import generate_gmm_data_fixed_means
from cblearn.embedding import SOE, CKL, GNMDS, FORTE, TSTE, MLDS
from questionnaire import Questionnaire
from estimators import OrdinalTangles
from matplotlib.cm import get_cmap
from triplets import majority_neighbours_count_matrix, subsample_triplets, triplets_to_majority_neighbour_cuts, unify_triplet_order
import pandas as pd

from tangles.data_types import Data

SEED = 1
RUNS_AVERAGED = 3
RESULTS_FOLDER = Path("results")


def evaluation_embedders(triplets: np.ndarray, responses: np.ndarray, embedding_dim: int, n_clusters: int, target: np.ndarray, seed: int) -> pd.DataFrame:
    """
    Evaluates the performance of a lot of baseline embedding algorithms
    on the given triplets and responses.
    """
    rows = []
    names = ["SOE", "CKL", "GNMDS", "FORTE", "TSTE", "MLDS"]
    embedders = [SOE(n_components=embedding_dim, random_state=seed), CKL(n_components=embedding_dim, random_state=seed),
                 GNMDS(n_components=embedding_dim, random_state=seed), FORTE(
        n_components=embedding_dim, random_state=seed), TSTE(n_components=embedding_dim, random_state=seed),
        MLDS(n_components=1, random_state=seed)]
    embeddings = [embedder.fit_transform(
        triplets, responses) for embedder in embedders]
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    preds = [kmeans.fit_predict(embedding) for embedding in embeddings]
    for i in range(len(embedders)):
        nmi = normalized_mutual_info_score(preds[i], target)
        ars = adjusted_rand_score(preds[i], target)
        name = names[i]
        rows.append(dict(method=name, nmi=nmi, ars=ars))

    return pd.DataFrame(rows)


def labels_to_colors(xs: np.ndarray) -> list[tuple]:
    cmap = get_cmap("tab10")
    return [cmap(x) for x in xs]


def plot_assignments(xs: np.ndarray, ys: np.ndarray):
    """
    Assumes contiguous labels y.
    """
    plt.figure()
    for i in range(np.max(ys.astype(int)) + 1):
        mask = (ys == i)
        plt.plot(xs[:, 0][mask], xs[:, 1][mask], ".")
    plt.xlabel("x")
    plt.ylabel("y")


def plot_line(df, x: str, y: str, methods_to_use: Optional[set[str]] = None):
    plt.figure()
    df = df.groupby(["method", x]).mean().reset_index()
    methods = set(df.method.unique())
    if methods_to_use is not None:
        methods = methods ^ methods_to_use
    for method in methods:
        x_arr = df[df.method == method][x]
        y_arr = df[df.method == method][y]
        plt.plot(x_arr, y_arr, "--o", label=f"{method}")
    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)


def plot_heatmap(df, x1: str, x2: str, y: str, method: str):
    plt.figure()
    df = df[df.method == method].groupby(
        [x1, x2]).mean().reset_index().sort_values([x1, x2])
    x1v, x2v = np.meshgrid(df[x1].unique(), df[x2].unique(), indexing="ij")
    yv = np.zeros_like(x1v)
    for i in range(x1v.shape[0]):
        for j in range(x1v.shape[1]):
            yv[i, j] = df[(df[x1] == x1v[i, j]) & (df[x2] == x2v[i, j])][y]
    plt.pcolormesh(x1v, x2v, yv, cmap="Blues")
    plt.colorbar()
    plt.clim(0.0, 1.0)
    plt.xlabel(x1)
    plt.ylabel(x2)
    plt.xlim([x1v.min(), x1v.max()])
    plt.ylim([x2v.min(), x2v.max()])


def save_plot(name: str):
    """
    Pass as name just the stem, no extension, no results folder appended.
    Saves the plot on the current graphical axis under the given name.
    """
    plt.savefig(RESULTS_FOLDER / f"{name}.pgf")
    plt.savefig(RESULTS_FOLDER / f"{name}.png")
    plt.savefig(RESULTS_FOLDER / f"{name}.pdf")


class TanglesMethod(Enum):
    EMBEDDING_FILL = "TANGLES-EMBEDDING-FILL"
    DIRECT = "TANGLES-DIRECT"
    MAJORITY_CUT = "TANGLES-MAJORITY"


def tangles_result_to_row(ys: np.ndarray, target: np.ndarray, tangles_method) -> pd.DataFrame:
    nmi = normalized_mutual_info_score(ys, target)
    ars = adjusted_rand_score(ys, target)
    return pd.DataFrame(dict(method=str(tangles_method), nmi=nmi, ars=ars), index=[0])


class Dataset(Enum):
    GAUSS_SMALL = 1
    GAUSS_LARGE = 2
    GAUSS_MASSIVE = 3

    @staticmethod
    def get(en: int, seed=SEED) -> Union[Data, tuple[np.ndarray, np.ndarray]]:
        """
        Returns the dataset described by the enum either as a Data object
        or as triplet-response combination (depends on dataset).
        """
        if en == Dataset.GAUSS_SMALL:
            means = np.array([[-6, 3], [-6, -3], [6, 3]])
            data = generate_gmm_data_fixed_means(20, means, std=1.0, seed=seed)
            return data
        elif en == Dataset.GAUSS_LARGE:
            means = np.array([[-6, 3], [-6, -3], [6, 3]])
            data = generate_gmm_data_fixed_means(
                200, means, std=0.7, seed=seed)
            return data
        else:
            raise ValueError(f"Dataset not supported (yet): {en}")


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
            #
            triplets, responses = q.to_bool_array(return_responses=True)
            df = evaluation_embedders(triplets, responses,
                                      embedding_dim=2, n_clusters=3, target=data.ys, seed=SEED + j).assign(density=density)
            dfs.append(df)

            tangles = OrdinalTangles(agreement=agreement)
            ys_tangles = tangles.fit_predict(q.values)
            dfs.append(tangles_result_to_row(
                ys_tangles, data.ys, TanglesMethod.DIRECT).assign(density=density))
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
                    triplets, responses, embedding_dim=2, n_clusters=3, target=data.ys, seed=SEED + j).assign(noise=noise, density=density)
                dfs.append(df)

                tangles = OrdinalTangles(agreement=8)
                ys_tangles = tangles.fit_predict(q.values)
                dfs.append(tangles_result_to_row(
                    ys_tangles, data.ys, TanglesMethod.DIRECT).assign(noise=noise, density=density))
    df = pd.concat(dfs, ignore_index=True).assign(
        experiment="sim-adding-noise-lowering-density-gauss")
    df.to_csv(RESULTS_FOLDER / "sim-noise-density.csv")
    for method in ["TANGLES-DIRECT", "SOE"]:
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


def main(experiment: str, debug: bool, n_runs: int):
    if experiment not in experiment_function_binding.keys():
        raise ValueError(f"Experiment {experiment} not implemented yet.")
    else:
        if debug:
            experiment_function_binding[experiment](n_runs=1, debug=True)
        else:
            experiment_function_binding[experiment](n_runs=n_runs, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the thesis experiments")
    parser.add_argument("experiment",
                        choices=available_experiments, help="Experiment to run")
    parser.add_argument(
        "--debug", action="store_true", help="Activates debug runs. Runs every experiment only once and with a reduced parameter set, if applicable. This is designed to make the experiments run fast as a priority.")
    parser.add_argument("--runs", type=int, default=RUNS_AVERAGED)
    args = parser.parse_args()
    main(args.experiment, args.debug, args.runs)
