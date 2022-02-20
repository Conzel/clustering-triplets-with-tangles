import unittest
from unittest import result
from sklearn.datasets import load_iris
from sklearn.metrics import normalized_mutual_info_score
from data_generation import generate_gmm_data_fixed_means
from estimators import OrdinalTangles
from experiment_runner import Questionnaire
from cblearn.datasets import make_random_triplets
from estimators import SoeKmeans
import pytest
from cblearn.embedding import SOE
from sklearn.cluster import KMeans
import numpy as np


def test_tangles_performance():
    synthetic_data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    tangles = OrdinalTangles(agreement=5, verbose=False)
    q = Questionnaire.from_metric(synthetic_data.xs, density=0.01, seed=1)
    tangles.fit(q.values)
    assert tangles.score(q.values, synthetic_data.ys) > 0.9


def test_soe_kmeans_performance():
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    q = Questionnaire.from_metric(data.xs, density=0.01, seed=1)
    soe_kmeans = SoeKmeans(embedding_dimension=2, n_clusters=5)
    pred = soe_kmeans.fit_predict(*q.to_bool_array())
    score = normalized_mutual_info_score(pred, data.ys)
    assert score > 0.95

    num_triplets = q.values.size
    # checking if performance is similar with Davids triplet generation
    t, r = make_random_triplets(
        data.xs, size=num_triplets, result_format="list-boolean")
    pred_david = soe_kmeans.fit_predict(t, r)
    score_david = normalized_mutual_info_score(pred_david, data.ys)
    assert score_david > 0.95

    assert np.abs(score - score_david) < 0.01
