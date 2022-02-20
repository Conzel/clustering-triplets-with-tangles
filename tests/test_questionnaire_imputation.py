from data_generation import generate_gmm_data_fixed_means
from questionnaire import Questionnaire
from sklearn.neighbors import DistanceMetric
from cblearn.datasets import fetch_car_similarity, make_random_triplets
from cblearn.utils import check_query_response
import numpy as np


def test_corruption_noise():
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    q = Questionnaire.from_metric(data.xs, noise=1.0)
    # In this case, all values should be corrupted
    assert np.all(q.values == -1)

    # corrupting half the values
    q = Questionnaire.from_metric(data.xs, noise=0.5)
    assert np.abs((q.values == -1).mean() - 0.5) < 0.01


def test_flip_noise():
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    q_flipped = Questionnaire.from_metric(data.xs, noise=1.0, flip_noise=True)
    q = Questionnaire.from_metric(data.xs)
    # In this case, all values should be flipped
    assert np.all(q_flipped.values != q.values)


def test_impute_random():
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    q = Questionnaire.from_metric(data.xs, noise=0.99)
    q = q.impute("random")
    assert np.abs(q.values.mean() - 0.5) < 0.01


def test_impute_knn():
    values = np.array([[1, 1, 1], [0, 0, 0], [1, 1, -1], [0, -1, 0]])
    labels = [(0, 1), (0, 2), (1, 2)]
    values_ = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])
    q = Questionnaire(values, labels)
    q = q.impute("1-NN")
    assert np.all(q.values == values_)


def test_impute_mean():
    values = np.array([[1, 1, 1], [0, 0, 0], [1, 1, -1], [0, -1, 0]])
    labels = [(0, 1), (0, 2), (1, 2)]
    values_ = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 0], [0, 1, 0]])
    q = Questionnaire(values, labels)
    q = q.impute("mean")
    assert np.all(q.values == values_)
