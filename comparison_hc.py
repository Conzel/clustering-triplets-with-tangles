"""
Class for the comparisonHC estimator. This will hopefully be included into cblearn at some point.
"""
import numpy as np
from typing import Optional
from comparisonhc import ComparisonHC as ComparisonHC_
from comparisonhc.oracle import OracleComparisons
from comparisonhc.linkage import OrdinalLinkageAverage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from hierarchies import HierarchyList
from triplets import reduce_triplets
from utils import flatten, index_cluster_list


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

        # if q[a, b, a, c] != 0 or q[a, c, a, b] != 0:
        #     raise ValueError(
        #         f"Unreduced triplets found (or responses): {t, r, i}")
        q[a, b, a, c] = 1
        q[a, c, a, b] = -1
    return q


class ComparisonHC():
    def __init__(self, num_clusters: int) -> None:
        self.num_clusters = num_clusters
        self._comparison_hc_original = None

    def fit_predict(self, triplets: np.ndarray, responses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        """
        # fit
        triplets = reduce_triplets(triplets, responses)
        quads = triplets_to_quadruplets(triplets)
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
        self._comparison_hc_original = chc
        self.labels_ = np.array(labels_for_original)
        return self.labels_

    def score(self, triplets: np.ndarray, responses: Optional[np.ndarray], ys: np.ndarray):
        return normalized_mutual_info_score(ys, self.fit_predict(triplets, responses))

    def aari(self, hierarchy: HierarchyList) -> float:
        """
        Returns the similarity between the two hierarchies according to the
        Average Adjusted Rand Index (see the appendix Ghoshdastidar et al., 2019
        for a more thorough definition).

        One argument is a hierarachy list, the other one is a dendrogram contained in an
        instance of the ComparisonHC class.
        For more, see https://github.com/mperrot/ComparisonHC/blob/3bed0d9d445c2c5a89fe0f9fb22047aa7b23960c/comparisonhc/core.py.
        """
        aris = []
        chc = self._comparison_hc_original
        assert chc is not None
        for l in range(1, hierarchy.depth + 1):
            clusters_left = hierarchy.clusters_at_level(l)
            if len(clusters_left) != 2**l:
                raise ValueError(
                    f"Hierarchy has {len(clusters_left)} clusters at level {l}, but should have {2**l}.")
            clusters_right = chc._get_k_clusters(
                chc.dendrogram, chc.clusters, 2**l)
            aris.append(adjusted_rand_score(index_cluster_list(
                clusters_left), index_cluster_list(clusters_right)))
        return np.mean(aris)


if __name__ == "__main__":
    from datasets import toy_gauss_triplets
    t, r, ys = toy_gauss_triplets(0.05)
    chc = ComparisonHC(7)
    chc.fit_predict(t, r)
