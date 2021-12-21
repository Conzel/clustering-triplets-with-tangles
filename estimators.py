"""
This module contains Scikit-Learn compatible estimators.
For more information on the API, refer to 
https://scikit-learn.org/stable/developers/develop.html
"""
import sys

from questionnaire import generate_questionnaire
sys.path.append("./tangles")

from sklearn.base import BaseEstimator
import src.data_types as data_types
import src.cost_functions as cost_functions
import src.utils as utils
import src.tree_tangles as tree_tangles
import numpy as np


class OrdinalTangles(BaseEstimator):
    def __init__(self, agreement=5, verbose=False):
        self.agreement = agreement
        self.verbose = verbose

    def fit(self, X, y=None):
        # Interpreting the questionnaires as cuts and computing their costs
        bipartitions = data_types.Cuts((X == 1).T)
        cost_function = cost_functions.BipartitionSimilarity(
            bipartitions.values.T)
        cuts = utils.compute_cost_and_order_cuts(
            bipartitions, cost_function, verbose=self.verbose)

        # Building the tree, contracting and calculating predictions
        tangles_tree = tree_tangles.tangle_computation(cuts=cuts,
                                                       agreement=self.agreement,
                                                       # print nothing
                                                       verbose=int(
                                                           self.verbose)
                                                       )

        contracted = tree_tangles.ContractedTangleTree(tangles_tree)
        contracted.prune(5, verbose=self.verbose)

        contracted.calculate_setP()
        self.contracted_tangles_tree_ = contracted

        # soft predictions
        weight = np.exp(-utils.normalize(cuts.costs))
        self.weight_ = weight

        return self

    def predict(self, X):
        bipartitions = data_types.Cuts((X == 1).T)
        tree_tangles.compute_soft_predictions_children(
            node=self.contracted_tangles_tree_.root, cuts=bipartitions, weight=self.weight_, verbose=self.verbose)

        ys_predicted, _ = utils.compute_hard_predictions(
            self.contracted_tangles_tree_, cuts=bipartitions, verbose=self.verbose)

        return ys_predicted
