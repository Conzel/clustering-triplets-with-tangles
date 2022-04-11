from sklearn.metrics import normalized_mutual_info_score
from comparison_hc import ComparisonHC, triplets_to_quadruplets
from data_generation import generate_gmm_data_fixed_means
import numpy as np
from triplets import reduce_triplets
from cblearn.datasets import make_random_triplets


def test_chc_performance():
    seed = 2
    data = generate_gmm_data_fixed_means(
        10, np.array([[1, 0], [-1, 0]]), 0.2, seed)
    chc = ComparisonHC(2)
    t = reduce_triplets(*make_random_triplets(data.xs,
                        result_format="list-boolean", size=5000, random_state=seed))
    y_chc = chc.fit_predict(t)
    assert normalized_mutual_info_score(y_chc, data.ys) > 0.99


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
