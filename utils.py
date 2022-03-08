import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from estimators import SoeKmeans


def compare_to_soe(triplets: np.ndarray, responses: np.ndarray, target: np.ndarray, ys_tangles: np.ndarray, embedding_dimension: int, n_clusters: int, seed=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduces some code duplication in the notebooks.

    Returns embedding and prediction of SOE.
    """
    soe_kmeans = SoeKmeans(
        embedding_dimension=embedding_dimension, n_clusters=n_clusters, seed=seed)
    ys_soe = soe_kmeans.fit_predict(triplets, responses)
    print(f"SOE-kMeans NMI: {normalized_mutual_info_score(ys_soe, target)}")
    print(
        f"Tangles NMI: {normalized_mutual_info_score(ys_tangles, target)} ({np.unique(ys_tangles).sum()})")
    return soe_kmeans.embedding_, ys_soe
