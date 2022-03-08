from data_generation import generate_gmm_data_fixed_means
from questionnaire import Questionnaire, generate_k_subsets, generate_question_set
from triplets import _lens_distance, lens_distance_matrix, triplets_to_majority_neighbour_cuts, subsample_triplets, unify_triplet_order, is_triplet
from sklearn.neighbors import DistanceMetric
from cblearn.datasets import make_random_triplets, make_all_triplets
from cblearn.utils import check_query_response
import numpy as np


def test_from_euclidean():
    q = Questionnaire.from_metric(np.array([[0, 0], [1, 1], [1, 1.5]]))
    assert (q.values == np.array([[1, 1, 1], [0, 0, 1], [0, 0, 0]])).all()


def test_from_manhattan():
    metric = DistanceMetric.get_metric('minkowski', p=1)
    q = Questionnaire.from_metric(
        np.array([[0, 0], [0, 1.6], [1, 1.01]]), metric=metric)
    # checking if the values are correct
    assert (q.values == np.array([[1, 1, 1], [0, 0, 1], [0, 0, 0]])).all()
    # checking if there is a meaningful difference to euclidean
    metric = DistanceMetric.get_metric('minkowski', p=2)
    q_euc = Questionnaire.from_metric(
        np.array([[0, 0], [0, 1.6], [1, 1]]), metric=metric)
    assert (q.values != q_euc.values).any()


def test_from_bipartitions():
    bipartitions = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])
    q = Questionnaire.from_bipartitions(bipartitions)
    assert np.all(q.values == bipartitions)
    assert set(q.labels) == set([(0, 1), (0, 2), (1, 2)])


def test_from_to_bool_array():
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    t, r = make_random_triplets(data.xs, "list-boolean")
    q = Questionnaire.from_bool_array(t, r)
    t_, r_ = q.to_bool_array()
    m = check_query_response(t, r, result_format="tensor-count")
    m_ = check_query_response(t_, r_, result_format="tensor-count")
    assert m == m_


def test_to_from_bool_array_full_triplets():
    # check if it also identical with all triplets
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    t, r = make_all_triplets(data.xs, "list-boolean")

    # checking if all triplets and only those have been used if no self fill
    q_no_self_fill = Questionnaire.from_bool_array(t, r, self_fill=False)
    assert (q_no_self_fill.values.size -
            (q_no_self_fill.values == -1).sum()) == t.shape[0]
    # checking if all values are identical if we fill
    q_self_fill = Questionnaire.from_bool_array(t, r, self_fill=True)
    q_ = Questionnaire.from_metric(data.xs)
    assert (q_self_fill.values == q_.values).all()


def test_questionnaire_subset():
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    q = Questionnaire.from_metric(data.xs)
    qs = q.subset(1000, seed=0)
    assert (qs.values != -1).sum() == 999


def test_self_label_fill():
    values = np.array([[-1, -1, 1], [-1, 0, -1], [1, -1, -1], [0, 1, 0]])
    labels = [(0, 1), (0, 2), (1, 2)]
    q = Questionnaire(values, labels)
    q_ = q.fill_self_labels()
    values_ = np.array([[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert (q_.values == values_).all()


def test_to_from_bool_array():
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    q = Questionnaire.from_metric(data.xs)
    t, r = q.to_bool_array()
    q_ = Questionnaire.from_bool_array(t, r)

    assert (q.values == q_.values).all()
    assert q.labels == q_.labels


def test_majority_cut():
    triplets = np.array([[0, 1, 2], [0, 1, 3], [1, 3, 2], [1, 3, 4]])
    cuts = triplets_to_majority_neighbour_cuts(triplets, radius=1)
    assert np.all(cuts == np.array([[1, 1, 0, 0, 0], [0, 1, 0, 1, 0]]).T)


def test_subsample():
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    q = Questionnaire.from_metric(data.xs)
    t, _ = subsample_triplets(data.xs, 100, return_responses=True)
    assert t.size == 300


def test_unify_triplet_order():
    t = unify_triplet_order(
        np.array([[0, 1, 2], [0, 3, 4], [1, 4, 6]]), np.array([True, False, True]))
    t_ = np.array([[0, 1, 2], [0, 4, 3], [1, 4, 6]])
    assert np.all(t == t_)


def test_lens_distance():
    t = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 3], [1, 3, 2]])
    # 0-1: 1
    # 0-2: 2
    # 2-3: 1
    r = np.array([1, 1, 2, 0])
    target_dists = np.array(
        [[0, 1, 2, 0], [1, 0, 0, 0], [2, 0, 0, 1], [0, 0, 1, 0]])
    assert np.all(target_dists == lens_distance_matrix(t, r))


def test_is_triplet():
    dists = np.array([[0, 2, 1], [2, 0, 4], [1, 4, 0]])
    assert is_triplet(0, 0, 1, dists)
    assert not is_triplet(1, 0, 1, dists)
    assert not is_triplet(1, 2, 0, dists)
    assert is_triplet(1, 0, 2, dists)


def test_subsets():
    assert generate_k_subsets([0, 1, 2], 2) == [[0, 1], [0, 2], [1, 2]]


def test_generate_question_set():
    assert generate_question_set(3) == [[0, 1], [0, 2], [1, 2]]
    assert generate_question_set(3, density=0.0) == set()
