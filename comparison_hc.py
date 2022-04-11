"""
Class for the comparisonHC estimator. This will hopefully be included into cblearn at some point.
"""
import numpy as np
from typing import Optional
from comparisonhc import ComparisonHC as ComparisonHC_
from comparisonhc.oracle import OracleComparisons
from comparisonhc.linkage import OrdinalLinkageAverage


def triplets_to_quadruplets(triplets: np.ndarray, responses: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Transforms an array of triplets (with responses) to an array of quadruplets.

    Assumes triplets, responses to be in list-boolean form, e.g. 

    responses[i] is True if triplets[i][0] is closer to triplets[i][1] than to 
    triplets[i][2].

    If responses is None, we assume that all responses are true (e.g. it is always triplets[i][1] closer).

    We return a quadruplet matrix that is filled according to the following scheme:
    If the triplet array allows for a statement (a,b,c) in triplet form then we
    set quadruplet[a,b,a,c] = 1.

    Triplets may contain duplicates or conflicting entries.
    In this case, we replace the value with a majority vote.
    """
    # error checking
    if len(triplets.shape) != 2:
        raise ValueError("Triplets must be a 2D array")
    if triplets.shape[1] != 3:
        raise ValueError("Triplets must have 3 columns")
    num_triplets = triplets.shape[0]
    if responses is None:
        responses = np.ones(num_triplets).astype(bool)
    if len(responses.shape) != 1:
        raise ValueError("Responses must be a 1D array or None")
    n = np.max(triplets) + 1
    q = np.zeros((n, n, n, n))

    for i in range(num_triplets):
        t = triplets[i]
        r = responses[i]
        if r:
            a, b, c = t[0], t[1], t[2]
        else:
            a, b, c = t[0], t[2], t[1]

        if q[a, b, a, c] != 0 or q[a, c, a, b] != 0:
            raise ValueError(
                f"Unreduced triplets found (or responses): {t, r, i}")
        q[a, b, a, c] = 1
        q[a, c, a, b] = -1
    return q


def flatten(l: list[list]) -> list:
    # sum can essentially be used as a mapReduce / flatMap ;)
    return sum(l, [])


class ComparisonHC():
    def __init__(self, num_clusters: int) -> None:
        self.num_clusters = num_clusters

    def fit_predict(self, triplets: np.ndarray, responses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        """
        # fit
        quads = triplets_to_quadruplets(triplets, responses)
        n = quads.shape[0]
        assert quads.shape == (n, n, n, n)
        oracle = OracleComparisons(quads)
        linkage = OrdinalLinkageAverage(oracle)
        chc = ComparisonHC_(linkage)
        chc.fit([[i] for i in range(n)])

        # predict
        clusters = chc._get_k_clusters(
            chc.dendrogram, chc.clusters, self.num_clusters)
        labels_in_order = flatten([[i] * len(cluster)
                                   for i, cluster in enumerate(clusters)])
        labels_for_original = [-1] * len(labels_in_order)
        for lab, pos in zip(labels_in_order, flatten(clusters)):
            labels_for_original[pos] = lab
        assert -1 not in labels_for_original
        return np.array(labels_for_original)
