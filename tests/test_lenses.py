from triplets import LensMetric
import numpy as np


def test_lens_distances_triplets():
    t = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 3], [1, 3, 2]])
    # 0-1: 1
    # 0-2: 2
    # 2-3: 1
    r = np.array([1, 1, 2, 0])
    target_dists = np.array(
        [[0, 1, 2, 0], [1, 0, 0, 0], [2, 0, 0, 1], [0, 0, 1, 0]])
    assert np.all(target_dists == LensMetric(
    ).pairwise_triplets(t, r, normalize=False))


def test_lens_metric_pairwise():
    xs = np.array([[2, 0], [-2, 0], [0, -1]])
    m = LensMetric()
    lens_dists = m.pairwise(xs)
    assert np.all(np.array([[0, 1, 0], [1, 0, 0], [
        0, 0, 0]]) == lens_dists)


def test_lenses_normalized():
    t = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 3], [1, 3, 2]])
    r = np.array([1, 1, 2, 0])
    target_dists = np.array(
        [[0, 1 / 3, 2 / 2, 0], [1 / 3, 0, 0, 0], [2 / 2, 0, 0, 1], [0, 0, 1, 0]])
    assert np.all(target_dists == LensMetric(
    ).pairwise_triplets(t, r, normalize=True))


def test_lens_metric_other_point():
    z = np.array([-3.0, 0.0])
    xs = np.array([[2, 0], [-2, 0], [0.1, 1], [0, -1]])
    m = LensMetric()
    dz0 = m.outside_point(xs, 0, z)
    assert dz0 == 3
    dz1 = m.outside_point(xs, 1, z)
    assert dz1 == 0
    dz2 = m.outside_point(xs, 2, z)
    assert dz2 == 2
    dz3 = m.outside_point(xs, 3, z)
    assert dz3 == 1
