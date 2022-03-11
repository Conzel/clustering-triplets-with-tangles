"""
File that is able to produce all the plots for the thesis.

Produce plot via calling the script and then calling
--exp <name>
"""
from __future__ import annotations
import argparse
from multiprocessing.sharedctypes import Value
from typing import Literal, Union
import numpy as np
from prometheus_client import Enum
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from data_generation import generate_gmm_data_fixed_means
from cblearn.embedding import SOE, CKL, GNMDS, FORTE, TSTE, MLDS
from questionnaire import Questionnaire
from estimators import OrdinalTangles
import pandas as pd

from tangles.data_types import Data

SEED = 1
RUNS_AVERAGED = 3


def evaluation_embedders(triplets: np.ndarray, responses: np.ndarray, embedding_dim: int, n_clusters: int, target: np.ndarray, n_runs: int = RUNS_AVERAGED) -> pd.DataFrame:
    """
    Evaluates the performance of a lot of baseline embedding algorithms
    on the given triplets and responses.
    """
    # classical methods
    rows = []
    names = ["SOE", "CKL", "GNMDS", "FORTE", "TSTE", "MLDS"]
    for j in range(n_runs):
        seed = SEED + j
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


class TanglesMethod(Enum):
    EMBEDDING_FILL = "TANGLES_EMBEDDING_FILL"
    DIRECT = "TANGLES_DIRECT"
    MAJORITY_CUT = "TANGLES_MAJORITY"


def tangles_result_to_row(ys: np.ndarray, target: np.ndarray, tangles_method) -> pd.DataFrame:
    nmi = normalized_mutual_info_score(ys, target)
    ars = adjusted_rand_score(ys, target)
    return pd.DataFrame(dict(method=str(tangles_method), nmi=nmi, ars=ars), index=[0])


class Dataset(Enum):
    GAUSS_SMALL = 1
    GAUSS_LARGE = 2

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


def simulation_all_triplets_gauss(n_runs=RUNS_AVERAGED):
    data = Dataset.get(Dataset.GAUSS_SMALL)
    q = Questionnaire.from_metric(data.xs)
    triplets, responses = q.to_bool_array(return_responses=True)
    df = evaluation_embedders(triplets, responses,
                              embedding_dim=2, n_clusters=3, target=data.ys, n_runs=n_runs)
    tangles = OrdinalTangles(agreement=8)
    ys_tangles = tangles.fit_predict(q.values)
    df = pd.concat((df, tangles_result_to_row(
        ys_tangles, data.ys, TanglesMethod.DIRECT)), ignore_index=True)
    return df.assign(experiment="sim-all-gauss")


def _sim_lower_density(data: Data, agreement: int, densities: list[float], n_runs: int) -> pd.DataFrame:
    dfs = []
    for density in densities:
        print(f"Density = {density}")
        q = Questionnaire.from_metric(data.xs, density=density, verbose=False)
        #
        triplets, responses = q.to_bool_array(return_responses=True)
        df = evaluation_embedders(triplets, responses,
                                  embedding_dim=2, n_clusters=3, target=data.ys, n_runs=n_runs).assign(density=density)
        dfs.append(df)

        tangles = OrdinalTangles(agreement=agreement)
        ys_tangles = tangles.fit_predict(q.values)
        dfs.append(tangles_result_to_row(
            ys_tangles, data.ys, TanglesMethod.DIRECT).assign(density=density))
    return pd.concat(dfs, ignore_index=True)


def simulation_lowering_density_gauss_small(n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    data = Dataset.get(Dataset.GAUSS_SMALL)
    densities = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]
    return _sim_lower_density(data, 8, densities, n_runs).assign(experiment="sim-lower-density-gauss-small")


def simulation_lowering_density_gauss_large(n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    data = Dataset.get(Dataset.GAUSS_LARGE)
    densities = [0.0005, 0.0001, 0.00008, 0.00005]
    return _sim_lower_density(data, 80, densities, n_runs).assign(experiment="sim-lower-density-gauss-large")


def simulation_adding_noise_gauss(n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    dfs = []
    data = Dataset.get(Dataset.GAUSS_SMALL)
    noises = [0.05, 0.1, 0.15, 0.2]
    for noise in noises:
        print(f"Noise = {noise}")
        q = Questionnaire.from_metric(
            data.xs, verbose=False, noise=noise, flip_noise=True)
        triplets, responses = q.to_bool_array(return_responses=True)
        df = evaluation_embedders(
            triplets, responses, embedding_dim=2, n_clusters=3, target=data.ys, n_runs=n_runs).assign(noise=noise)
        dfs.append(df)

        tangles = OrdinalTangles(agreement=8)
        ys_tangles = tangles.fit_predict(q.values)
        dfs.append(tangles_result_to_row(
            ys_tangles, data.ys, TanglesMethod.DIRECT).assign(noise=noise))

    return pd.concat(dfs, ignore_index=True).assign(experiment="sim-adding-noise-gauss")


def simulation_adding_noise_and_lowering_density(n_runs=RUNS_AVERAGED) -> pd.DataFrame:
    dfs = []
    noises = [0.05, 0.1, 0.15, 0.2]
    data = Dataset.get(Dataset.GAUSS_SMALL)
    densities = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001]
    for noise in noises:
        for density in densities:
            print(f"Density = {density}, Noise = {noise}")
            q = Questionnaire.from_metric(
                data.xs, density=density, verbose=False, noise=noise, flip_noise=True)
            triplets, responses = q.to_bool_array(return_responses=True)
            df = evaluation_embedders(
                triplets, responses, embedding_dim=2, n_clusters=3, target=data.ys, n_runs=n_runs).assign(noise=noise, density=density)
            dfs.append(df)

            tangles = OrdinalTangles(agreement=8)
            ys_tangles = tangles.fit_predict(q.values)
            dfs.append(tangles_result_to_row(
                ys_tangles, data.ys, TanglesMethod.DIRECT).assign(noise=noise, density=density))
    return pd.concat(dfs, ignore_index=True).assign(experiment="sim-adding-noise-lowering-density-gauss")


available_experiments = ["sim-all-gauss"]

experiment_function_binding = {
    "sim-all-gauss": simulation_all_triplets_gauss,
}


def main(experiment: str):
    if experiment not in experiment_function_binding.keys():
        raise ValueError(f"Experiment {experiment} not implemented yet.")
    else:
        experiment_function_binding[experiment]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the thesis experiments")
    parser.add_argument("experiment",
                        choices=available_experiments, help="Experiment to run")
    args = parser.parse_args()
    main(args.experiment)
