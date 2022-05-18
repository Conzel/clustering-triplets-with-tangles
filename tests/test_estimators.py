from sklearn.metrics import normalized_mutual_info_score
from data_generation import generate_gmm_data_fixed_means
from estimators import LandmarkTangles, MajorityTangles, OrdinalTangles
from cblearn.datasets import make_random_triplets
from estimators import SoeKmeans
import numpy as np
from questionnaire import Questionnaire
from tangles.tree_tangles import get_hard_predictions
from datasets import Dataset


def test_tangles_performance():
    synthetic_data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    tangles = OrdinalTangles(agreement=5, verbose=False)
    q = Questionnaire.from_metric(synthetic_data.xs, density=0.01, seed=1)
    tangles.fit(q.values)
    assert normalized_mutual_info_score(
        tangles.labels_, synthetic_data.ys) > 0.9


def test_landmark_tangles_performance():
    synthetic_data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    tangles = LandmarkTangles(agreement=5, verbose=False)
    t, r = Questionnaire.from_metric(
        synthetic_data.xs, density=0.01, seed=1).to_bool_array()
    assert tangles.score(t, r, synthetic_data.ys) > 0.9


def test_majority_cut_tangles_performance():
    synthetic_data = generate_gmm_data_fixed_means(
        n=10, means=np.array(np.array([[-1, 0], [1, 0]])), std=0.2, seed=1)
    tangles = MajorityTangles(agreement=4, verbose=False)
    t, r = Questionnaire.from_metric(
        synthetic_data.xs, density=1, seed=1).to_bool_array()
    assert tangles.score(t, r, synthetic_data.ys) > 0.9


def test_soe_kmeans_performance():
    data = generate_gmm_data_fixed_means(
        n=5, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.3, seed=1)
    q = Questionnaire.from_metric(data.xs, density=0.1, seed=1)
    soe_kmeans = SoeKmeans(embedding_dimension=2, n_clusters=5, seed=1)
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


def test_soe_kmeans_silhouette_performance():
    # We have to go a bit easy on the Silhouette method.
    # With the values above, it does not work very well
    # (which is interesting... maybe this could be used for our purposes)
    data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[10, -10], [-9, 7]])), std=0.1, seed=1)
    q = Questionnaire.from_metric(data.xs, density=0.1, seed=2)
    soe_kmeans = SoeKmeans(embedding_dimension=2, n_clusters=None, k_max=5)
    pred = soe_kmeans.fit_predict(*q.to_bool_array())
    score = normalized_mutual_info_score(pred, data.ys)
    assert score > 0.95
    assert soe_kmeans.k_ == 2


def test_estimator_same_as_tangles_impl():
    """
    Ensures the estimator performs the same as the implementation
    in the tangles module.
    """
    synthetic_data = generate_gmm_data_fixed_means(
        n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
    tangles = OrdinalTangles(agreement=5, verbose=False)
    q = Questionnaire.from_metric(synthetic_data.xs, density=0.01, seed=1)
    tangles.fit(q.values)
    assert np.all(tangles.labels_ == get_hard_predictions(q.values, 5))


def test_tangles_predict_hierarchy():
    data = Dataset.get(Dataset.GAUSS_SMALL, seed=0)
    q = Questionnaire.from_metric(data.xs)
    tangles = OrdinalTangles(agreement=5, verbose=False)
    tangles.fit(q.values)
    assert tangles.predict_hierarchy(
    ) == [[list(range(20)), list(range(20, 40))], list(range(40, 60))]
