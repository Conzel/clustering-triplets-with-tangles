"""
This module contains Scikit-Learn compatible estimators.
For more information on the API, refer to 
https://scikit-learn.org/stable/developers/develop.html
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils.validation import check_is_fitted

from tangles.cost_functions import BipartitionSimilarity
from tangles.data_types import Cuts
from tangles.tree_tangles import (ContractedTangleTree,
                                  compute_soft_predictions_children,
                                  tangle_computation)
from tangles.utils import compute_cost_and_order_cuts, compute_hard_predictions, normalize


class OrdinalTangles(BaseEstimator):
    def __init__(self, agreement=5, verbose=False):
        self.agreement = agreement
        self.verbose = verbose

    def fit(self, X, y=None):
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
        # contracted.prune(1, verbose=self.verbose)

        contracted.calculate_setP()

        # soft predictions
        weight = np.exp(-normalize(cuts.costs))

        self.weight_ = weight
        self.contracted_tangles_tree_ = contracted

        return self

    def predict(self, X):
        check_is_fitted(self, ["weight_", "contracted_tangles_tree_"])
        bipartitions = Cuts((X == 1).T)

        # Here we have to reorder the cuts... this is pricey,
        # maybe we can prevent it.
        cost_function = BipartitionSimilarity(
            bipartitions.values.T)
        # cuts = utils.compute_cost_and_order_cuts(
        #     bipartitions, cost_function, verbose=self.verbose)
        cuts = bipartitions

        compute_soft_predictions_children(
            node=self.contracted_tangles_tree_.root, cuts=cuts, weight=self.weight_, verbose=self.verbose)

        ys_predicted, _ = compute_hard_predictions(
            self.contracted_tangles_tree_, cuts=bipartitions, verbose=self.verbose)

        return ys_predicted

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError(
            "Soft predictions have not been implemented yet.")

    def score(self, X, y):
        y_pred = self.predict(X)
        return normalized_mutual_info_score(y, y_pred)
