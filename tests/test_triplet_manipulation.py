from triplets import unify_triplet_order, unify_triplets_mostcentral, reduce_triplets, reduce_triplets_mostcentral
import numpy as np


def test_normal_triplets_unify():
    triplets = np.array(
        [[0, 1, 2], [1, 2, 3], [0, 2, 1], [0, 1, 2], [3, 2, 1]])
    responses = np.array([0, 1, 0, 1, 0])
    expected = np.array(
        [[0, 2, 1], [1, 2, 3], [0, 1, 2], [0, 1, 2], [3, 1, 2]])
    assert np.all(unify_triplet_order(triplets, responses) == expected)


def test_normal_triplets_reduce():
    triplets = np.array(
        [[0, 1, 2], [1, 2, 3], [0, 2, 1], [0, 1, 2], [3, 2, 1]])
    responses = np.array([0, 1, 1, 1, 1])
    assert np.all(reduce_triplets(triplets, responses) == np.array(
        [[0, 2, 1], [1, 2, 3], [3, 2, 1]]))
    assert np.all(reduce_triplets(triplets) == np.array(
        [[0, 1, 2], [1, 2, 3], [3, 2, 1]]))
    assert np.all(reduce_triplets(triplets, np.array([1, 1, 1, 1, 1])) == np.array(
        [[0, 1, 2], [1, 2, 3], [3, 2, 1]]))


def test_mostcentral_triplets_unify():
    triplets = np.array(
        [[0, 1, 2], [1, 2, 3], [0, 2, 1], [0, 1, 2], [3, 2, 1]])
    responses = np.array([0, 2, 2, 0, 2])
    assert np.all(unify_triplets_mostcentral(triplets, responses) == np.array(
        [[0, 1, 2], [3, 1, 2], [1, 0, 2], [0, 1, 2], [1, 3, 2]]))


def test_mostcentral_triplets_reduce():
    triplets = np.array(
        [[0, 1, 2], [1, 2, 3], [0, 2, 1], [0, 1, 2], [3, 2, 1]])
    responses = np.array([0, 2, 2, 0, 2])
    assert np.all(reduce_triplets_mostcentral(triplets, responses) == np.array(
        [[0, 1, 2], [3, 1, 2]]))
