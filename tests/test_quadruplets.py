from comparison_hc import triplets_to_quadruplets
import numpy as np


def test_quadruplet_generation():
    triplets = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    responses = np.array([1, 1, 0])
    quad_responses = triplets_to_quadruplets(triplets, responses)
    assert quad_responses[0, 1, 0, 2] == 1
    assert quad_responses[1, 2, 1, 3] == 1
    assert quad_responses[2, 3, 2, 4] == -1
    assert quad_responses[0, 2, 0, 1] == -1
    assert quad_responses[1, 3, 1, 2] == -1
    assert quad_responses[2, 4, 2, 3] == 1

    quad_no_responses = triplets_to_quadruplets(triplets)
    assert quad_no_responses[0, 1, 0, 2] == 1
    assert quad_no_responses[1, 2, 1, 3] == 1
    assert quad_no_responses[2, 3, 2, 4] == 1
    assert quad_no_responses[0, 2, 0, 1] == -1
    assert quad_no_responses[1, 3, 1, 2] == -1
    assert quad_no_responses[2, 4, 2, 3] == -1

    assert (quad_responses == 1).sum() == 3
    assert (quad_responses == -1).sum() == 3
    assert quad_responses.sum() == 0

    assert (quad_no_responses == 1).sum() == 3
    assert (quad_no_responses == -1).sum() == 3
    assert quad_no_responses.sum() == 0


def test_quadruplets_symmetric():
    triplets = np.array([[0, 1, 2]])
    q = triplets_to_quadruplets(triplets, responses=None, symmetry=True)
    assert q[0, 1, 0, 2] == 1
    assert q[0, 1, 2, 0] == 1
    assert q[1, 0, 2, 0] == 1
    assert q[1, 0, 0, 2] == 1

    assert q[0, 2, 0, 1] == -1
    assert q[2, 0, 0, 1] == -1
    assert q[2, 0, 1, 0] == -1
    assert q[0, 2, 1, 0] == -1

    assert q.sum() == 0
    assert (q == 1).sum() == 4
    assert (q == -1).sum() == 4
