#!/usr/bin/env python3
import numpy as np
import sklearn
from tangles.data_types import Data
from tangles.loading import load_GMM


def draw_cluster_means(num_clusters, dimension, minimum_cluster_distance):
    """
    Draws cluster centers for the gaussian mixture model. The cluster centers
    are drawn according to a multidimensional gaussian. The cluster centers
    are then scaled so that the minimal distance between two clusters is equal
    to minimum_cluster_distance.

    Input:
        num_clusters: int,
            number of cluster centers
        dimension: int,
            dimension of the cluster centers
        minimum_cluster_distance: float,
            minimum distance that two cluster centers have 
    output: 
        ndarray of dimension [num_clusters, dimension],
        containing the cluster centers for each cluster
    """
    cluster_centers = np.random.randn(num_clusters, dimension)
    return rescale_points(cluster_centers, minimum_cluster_distance)


def rescale_points(x, desired_min_dist):
    """
    Scales an array x of points such that they have at least the given distance to each other.
    """
    if desired_min_dist > 0:
        dists = sklearn.metrics.pairwise_distances(x)
        current_min_dist = dists[~np.eye(dists.shape[0], dtype=bool)].min()
        return x / current_min_dist * desired_min_dist
    else:
        return x


def generate_gmm_data_fixed_means(n: int, means: np.ndarray, std: float, seed: int) -> Data:
    return _generate_gmm_data(n, means, std, seed)


def generate_gmm_data_draw_means(n: int, std: float, seed: int, components: int, dimension: int, min_cluster_dist: float) -> data_types.Data:
    """
    Generates Data according to a gaussian mixture model with the parameters.

    n: Number of sample points per component
    means: ndarray of dimension [num_clusters, dimension], or None (if it should be drawn dynamically)
    """
    means = draw_cluster_means(components, dimension, min_cluster_dist)
    return _generate_gmm_data(n, means, std, seed)


def _generate_gmm_data(n, means: np.ndarray, std: float, seed: int):
    stds = std * np.ones(means.shape)
    num_clusters = len(means)

    xs, ys = load_GMM(blob_sizes=[n] * num_clusters,
                      blob_centers=means, blob_variances=stds, seed=seed)
    data = Data(xs=xs, ys=ys)
    return data
