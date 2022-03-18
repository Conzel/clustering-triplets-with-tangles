"""
This module contains Scikit-Learn compatible estimators.
For more information on the API, refer to 
https://scikit-learn.org/stable/developers/develop.html
"""

from random import random
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.utils.validation import check_is_fitted
from cblearn.embedding import SOE

from tangles.cost_functions import BipartitionSimilarity
from tangles.data_types import Cuts
from tangles.tree_tangles import (ContractedTangleTree,
                                  compute_soft_predictions_children,
                                  tangle_computation)
from tangles.utils import compute_cost_and_order_cuts, compute_hard_predictions, normalize


class OrdinalTangles(BaseEstimator):
    def __init__(self, agreement=5, verbose=0):
        """
        Initializes a tangles clustering algorithm for triplet data.

        verbose: Set to 0 for no console output, 
                 1 for status updates, 2 for debugging
                 and 3 for inspection (producing plots of cuts etc.)
        """
        self.agreement = agreement
        self.verbose_level = verbose > 0
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Does nothing as the model is an unsupervised algorithm. 
        """
        return self

    def predict(self, X):
        # Interpreting the questionnaires as cuts and computing their costs
        if not np.all(np.logical_or(X == 0, X == 1)):
            raise ValueError(
                "X contains illegal values. X must only contain values equal to 0 or 1. You might have forgotten to impute missing values?")
        bipartitions = Cuts((X == 1).T)
        cost_function = BipartitionSimilarity(
            bipartitions.values.T)
        cuts = compute_cost_and_order_cuts(
            bipartitions, cost_function, verbose=self.verbose)

        # Building the tree, contracting and calculating predictions
        tangles_tree = tangle_computation(cuts=cuts,
                                          agreement=self.agreement,
                                          # print nothing
                                          verbose=int(
                                              self.verbose)
                                          )

        contracted = ContractedTangleTree(tangles_tree)
        contracted.prune(1, verbose=self.verbose)

        contracted.calculate_setP()

        # soft predictions
        weight = np.exp(-normalize(cuts.costs))

        self.weight_ = weight
        self.contracted_tangles_tree_ = contracted
        self.tangles_tree_ = tangles_tree

        bipartitions = Cuts((X == 1).T)

        cost_function = BipartitionSimilarity(
            bipartitions.values.T)
        _ = compute_cost_and_order_cuts(
            bipartitions, cost_function, verbose=self.verbose)

        compute_soft_predictions_children(
            node=self.contracted_tangles_tree_.root, cuts=bipartitions, weight=weight, verbose=self.verbose)
        contracted.processed_soft_predictions = True

        ys_predicted, _ = compute_hard_predictions(
            self.contracted_tangles_tree_, verbose=self.verbose)

        self.cuts_ = bipartitions
        return ys_predicted

    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError(
            "Soft predictions have not been implemented yet.")

    def score(self, X, y):
        y_pred = self.fit_predict(X)
        return normalized_mutual_info_score(y, y_pred)


class SoeKmeans(BaseEstimator):
    def __init__(self, embedding_dimension, n_clusters, seed=None, k_max=20):
        """
        Initializes a classifier that uses a combination of SOE and kMeans
        to predict labels of given triplets.

        First, an embedding is created via SOE. Next, the embedding is
        used to run a k-Means cluster on it.

        If n_clusters is explicitly set to None, the Silhouette method will
        be used to infer k. k_max will then be the maximum number of k
        to search up to.

        Input is triplets, responses as in the detailed in cblearn.
        """
        self.embedding_dimension = embedding_dimension
        self.n_clusters = n_clusters
        self.seed = seed
        self._soe = SOE(n_components=self.embedding_dimension,
                        random_state=self.seed)
        if n_clusters is None:
            self._kmeans = None
            self._k_max = k_max
            self._k = None
        else:
            self._kmeans = KMeans(n_clusters=self.n_clusters,
                                  random_state=self.seed)
            self._k = n_clusters
        self._embedding = None

    @property
    def k_(self):
        if self._k is None:
            raise ValueError(
                "No value of k has been determined yet. Call fit_predict (uses Silhouette method to choose k).")
        else:
            return self._k

    @property
    def embedding_(self):
        if self._embedding is None:
            raise ValueError("Not fit yet. Call fit first.")
        else:
            return self._embedding

    def fit(self, triplets, responses, y=None):
        """
        Use fit_predict directly.
        """
        return self

    def predict(self, triplets, responses):
        """
        Use fit_predict directly.
        """
        return self

    def fit_predict(self, triplets, responses, y=None) -> np.ndarray:
        """
        Performs SOE-kMeans to predict labels of given triplets.
        """
        self._embedding = self._soe.fit_transform(triplets, responses)
        if self._kmeans is None:
            k = find_k_silhouette(self.embedding_, k_max=self._k_max)
            self._k = k
            kmeans = KMeans(n_clusters=k, random_state=self.seed)
            self._kmeans = kmeans
        ys = self._kmeans.fit_predict(self.embedding_)
        return ys

    def score(self, triplets, responses, y):
        """
        Returns clustering score via NMI.
        """
        y_pred = self.fit_predict(triplets, responses)
        return normalized_mutual_info_score(y, y_pred)


def find_k_silhouette(xs: np.ndarray, k_max: int = 20) -> int:
    """
        When using, keep in mind that k_max of 20 might not be necessarily desirable

        We use the Silhouette method of finding an optimal k as a starter,
        since it's pretty easy. Different methods of finding optimal k might be gleaned
        from von Luxburg: https://arxiv.org/abs/1007.1075

        Blog article where the silhouette method is described:
        https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    """
    sil = []
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    ks = list(range(2, k_max + 1))

    for k in ks:
        kmeans = KMeans(n_clusters=k).fit(xs)
        labels = kmeans.labels_
        sil.append(silhouette_score(xs, labels, metric='euclidean'))

    optimal_k = ks[np.argmax(sil)]
    return optimal_k
