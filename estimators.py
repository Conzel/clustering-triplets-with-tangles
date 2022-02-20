"""
This module contains Scikit-Learn compatible estimators.
For more information on the API, refer to 
https://scikit-learn.org/stable/developers/develop.html
"""

from random import random
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
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
        return ys_predicted

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError(
            "Soft predictions have not been implemented yet.")

    def score(self, X, y):
        y_pred = self.fit_predict(X)
        return normalized_mutual_info_score(y, y_pred)


class SoeKmeans(BaseEstimator):
    def __init__(self, embedding_dimension, n_clusters, seed=None):
        """
        Initializes a classifier that uses a combination of SOE and kMeans
        to predict labels of given triplets.

        First, an embedding is created via SOE. Next, the embedding is
        used to run a k-Means cluster on it.

        Input is triplets, responses as in the detailed in cblearn.
        """
        self.embedding_dimension = embedding_dimension
        self.n_clusters = n_clusters
        self.seed = seed
        self._soe = SOE(n_components=self.embedding_dimension,
                        random_state=self.seed)
        self._kmeans = KMeans(n_clusters=self.n_clusters,
                              random_state=self.seed)
        self._embedding = None

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

    def fit_predict(self, triplets, responses, y=None):
        """
        Performs SOE-kMeans to predict labels of given triplets.
        """
        self._embedding = self._soe.fit_transform(triplets, responses)
        ys = self._kmeans.fit_predict(self._soe.embedding_)
        return ys

    def score(self, triplets, responses, y):
        """
        Returns clustering score via NMI.
        """
        y_pred = self.fit_predict(triplets, responses)
        return normalized_mutual_info_score(y, y_pred)
