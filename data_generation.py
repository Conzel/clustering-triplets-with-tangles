#!/usr/bin/env python3
"""
Module for generating synthetic data. Current possible data sets are:
- Gaussian Mixture Model
- Stochastic Block Model
"""
from typing import Optional, Union
import numpy as np
import sklearn
import networkx as nx
from torchvision.datasets import USPS
from utils import flatten

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


def generate_smb_data(n: int, k: int, p: float, q: float) -> tuple[nx.Graph, np.ndarray]:
    """
    Very simple SMB, all k blocks have the same size n, 
    an edge goes from i to j with probability p if i == j,
    and with probability q if i != j.

    Output: 
    """
    assert p > 0 and p < 1
    sizes = [n] * k
    labels = []
    for i in range(k):
        labels.extend([i] * n)
    probabilities = [[p if i == j else q for i in range(k)] for j in range(k)]
    return nx.stochastic_block_model(sizes, probabilities), np.array(labels)


def clean_bipartitions(xs: np.ndarray) -> np.ndarray:
    """
    Takes in bipartitions (n_datapoints, n_cuts) and
    returns another set of bipartitions with degenerate cuts removed
    (all 0 or all 1).
    """
    non_zero = np.logical_not(np.all(xs == 0, axis=0))
    non_one = np.logical_not(np.all(xs == 1, axis=0))
    res = xs[:, np.logical_and(non_zero, non_one)]
    return res


def generate_gmm_data_fixed_means(n: Union[int, list[int]], means: np.ndarray, std: float, seed: int) -> Data:
    if isinstance(n, list):
        xs = []
        ys = []
        i = 0
        for cluster_size, mean in zip(n, means):
            data_n = _generate_gmm_data(
                cluster_size, mean[None, :], std, seed + i)
            xs.append(data_n.xs)
            ys.append(data_n.ys + i)
            i += 1
        return Data(xs=np.concatenate(xs), ys=np.concatenate(ys))
    return _generate_gmm_data(n, means, std, seed)


def generate_gmm_data_draw_means(n: int, std: float, seed: int, components: int, dimension: int, min_cluster_dist: float) -> Data:
    """
    Generates Data according to a gaussian mixture model with the parameters.

    n: Number of sample points per component
    means: ndarray of dimension [num_clusters, dimension], or None (if it should be drawn dynamically)
    """
    means = draw_cluster_means(components, dimension, min_cluster_dist)
    return _generate_gmm_data(n, means, std, seed)


def get_usps(shuffle: bool = True, seed: Optional[int] = None, subset: Optional[set[int]] = None, num_samples: Optional[int] = None) -> Data:
    """
    Returns data from the USPS dataset. Values returned are flattened arrays of the images with values in [0,255],
    as well as the number that the image depicts. 

    If subset is set to a set of ints, returns only those numbers.
    If num_samples is set, returns this many samples per number. Only useable if subset is not None.
    """
    if num_samples is not None and subset is None:
        raise ValueError(
            "Illegal parameter combination: num_samples is set but subset is None")
    usps = USPS(root='./datasets', download=True)
    labels = []
    images = []
    for im, label in usps:
        images.append(np.asarray(im).flatten()[None, :])
        labels.append(label)
    im_arr = np.concatenate(images, axis=0)
    label_arr = np.array(labels)
    if subset is not None:
        is_in_subset = [l in subset for l in labels]
        im_arr = im_arr[is_in_subset, :]
        label_arr = label_arr[is_in_subset]

    if shuffle:
        np.random.seed(seed)
        perm = np.random.permutation(im_arr.shape[0])
        im_arr = im_arr[perm, :]
        label_arr = label_arr[perm]

    if num_samples is not None and subset is not None:
        im_arr_subsampled = np.zeros(
            (num_samples * len(subset), im_arr.shape[1]))
        label_arr_subsampled = np.zeros(num_samples * len(subset))
        for (k, j) in enumerate(subset):
            subsample = np.where(label_arr == j)[0][:num_samples]
            im_arr_subsample = im_arr[subsample, :]
            im_arr_subsampled[(k * num_samples):((k + 1) *
                                                 num_samples), :] = im_arr_subsample
            label_arr_subsampled[(k * num_samples):((k + 1) * num_samples)] = j

        perm = np.random.permutation(im_arr_subsampled.shape[0])
        im_arr = im_arr_subsampled[perm, :]
        label_arr = label_arr_subsampled[perm]
    return Data(im_arr, label_arr)


def _generate_gmm_data(n, means: np.ndarray, std: float, seed: int):
    stds = std * np.ones(means.shape)
    num_clusters = len(means)

    xs, ys = load_GMM(blob_sizes=[n] * num_clusters,
                      blob_centers=means, blob_variances=stds, seed=seed)
    data = Data(xs=xs, ys=ys)
    return data


def generate_planted_hierarchy(num_classes_exp: int, num_per_class: int, initial_class_dist: float, class_dist_sim_decrease: float = 1, noise_variance: float = 0) -> Data:
    """
    Generated according to Ghoshdastidar et al., 2019, Foundations of Comparison-Based Hierarchical Clustering.
    This model is represented as a similarity matrix S and
    corresponds to a noisy hierarchical block matrix, where S = M + R, 
    with R being a symmetric perturbation matrix.
    """
    L = num_classes_exp
    N0 = num_per_class
    N = 2**L * N0
    mu = initial_class_dist
    delta = class_dist_sim_decrease
    M = np.zeros((N, N))

    def _set_sims(sims, start, end, l):
        middle = int((end - start) / 2) + start
        if l == L:
            return
        off_diag_value = mu - (L - l) * delta
        sims[middle:end, start:middle] = off_diag_value
        sims[start:middle, middle:end] = off_diag_value
        _set_sims(sims, start, middle, l + 1)
        _set_sims(sims, middle, end, l + 1)

    R = np.random.normal(0, noise_variance, (N, N))
    M[M == 0] = mu
    _set_sims(M, 0, N, 0)
    return Data(xs=M + R, ys=flatten([[i] * N0 for i in range(2**L)]))
