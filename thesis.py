"""
This module contains utility functions and classes that help in creating
the plots shown in the thesis.
"""
from __future__ import annotations
import os
from typing import Optional
from pathlib import Path
from comparison_hc import ComparisonHC
import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline
from cblearn.embedding import SOE, CKL, GNMDS, FORTE, TSTE, MLDS
from hierarchies import BinaryHierarchyTree, aari
from questionnaire import Questionnaire
from estimators import LandmarkTangles, MajorityTangles, SoeKmeans
import pandas as pd
from data_generation import generate_planted_hierarchy


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


def eval_hierarchical(noise=0.0, density=0.1, hier_noise=0.0, n_runs=1):
    """
    Evaluates a simple hierarchical block matrix and returns the mean of the results
    (landmark, soe, majority, comparison, hierarchical landmarks, hierarchical comparisonhc)
    """
    l, s, m, c, hl, hc = [], [], [], [], [], []
    true_hierarchy = BinaryHierarchyTree([[list(range(0, 10)), list(range(10, 20))], [
                                         list(range(20, 30)), list(range(30, 40))]])
    for i in range(n_runs):
        data = generate_planted_hierarchy(2, 10, 5, 1, hier_noise)
        q = Questionnaire.from_precomputed(
            data.xs, density=density, use_similarities=True, noise=noise, verbose=False).impute("random")
        t, r = q.to_bool_array()
        chc = ComparisonHC(4)
        y_chc = chc.fit_predict(t, r)
        lt = LandmarkTangles(4).fit(t, r, data.ys)
        m.append(MajorityTangles(4, radius=1 / 2).score(t, r, data.ys))
        l.append(LandmarkTangles(4).score(t, r, data.ys))
        s.append(SoeKmeans(2, 4).score(t, r, data.ys))
        c.append(ComparisonHC(4).score(t, r, data.ys))
        hl.append(aari(true_hierarchy, BinaryHierarchyTree(lt.hierarchy_), 2))
        hc.append(aari(true_hierarchy, chc, 2))
        #hm.append(aari(true_hierarchy))

    return np.mean(l), np.mean(s), np.mean(m), np.mean(c), np.mean(hl), np.mean(hc)
