from sklearn.metrics import normalized_mutual_info_score
from comparison_hc import ComparisonHC
from data_generation import generate_gmm_data_fixed_means, generate_planted_hierarchy
from estimators import EmbedderHierarchicalClustering, LandmarkTangles, MajorityTangles, OrdinalTangles
from hierarchies import BinaryHierarchyTree, aari
from triplets import reduce_triplets
from cblearn.datasets import make_random_triplets
from cblearn.embedding import SOE
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


def test_tangles_predict_gauss_hierarchy_exact_reconstruction():
    data = Dataset.get(Dataset.GAUSS_SMALL, seed=0)
    q = Questionnaire.from_metric(data.xs)
    tangles = OrdinalTangles(agreement=5, verbose=False)
    tangles.fit(q.values)
    assert tangles.hierarchy_ == [
        [list(range(20)), list(range(20, 40))], list(range(40, 60))]


def between(a, b):
    return list(range(a, b))


def test_comparison_hc_planted_hierarchy_performance():
    data = generate_planted_hierarchy(2, 5, 0.8, 2.0, 0)
    q = Questionnaire.from_precomputed(
        data.xs, density=1.0, use_similarities=True)
    t, r = q.to_bool_array()
    chc = ComparisonHC(4)
    ys = chc.fit_predict(t, r)
    score = normalized_mutual_info_score(ys, data.ys)
    assert score > 0.99
    hierarchy_truth_list = [[between(0, 5), between(5, 10)], [
        between(10, 15), between(15, 20)]]
    assert aari(chc, BinaryHierarchyTree(hierarchy_truth_list), 2) == 1.0


def test_soe_al_planted_hierarchy_performance():
    data = generate_planted_hierarchy(2, 5, 5, 1, 0)
    q = Questionnaire.from_precomputed(
        data.xs, density=1.0, use_similarities=True)
    t, r = q.to_bool_array()
    soe_al = EmbedderHierarchicalClustering(
        SOE(2, random_state=0), 4, linkage="average")
    ys = soe_al.fit_predict(t, r)
    score = normalized_mutual_info_score(ys, data.ys)
    assert score > 0.95
    hierarchy_truth_list = [[between(0, 5), between(5, 10)], [
        between(10, 15), between(15, 20)]]
    assert aari(soe_al, BinaryHierarchyTree(
        hierarchy_truth_list), 2) >= 0.5  # its bad


def test_comparison_hc_gauss_clustering_performance():
    seed = 2
    data = generate_gmm_data_fixed_means(
        10, np.array([[1, 0], [-1, 0]]), 0.2, seed)
    chc = ComparisonHC(2)
    t = reduce_triplets(*make_random_triplets(data.xs,
                        result_format="list-boolean", size=5000, random_state=seed))
    y_chc = chc.fit_predict(t)
    assert normalized_mutual_info_score(y_chc, data.ys) > 0.99


def test_tangles_predict_planted_hierarchy_performance():
    data = generate_planted_hierarchy(2, 10, 0.8, 2.0, 0)
    q = Questionnaire.from_precomputed(data.xs, density=0.1)
    t, r = q.to_bool_array()
    tangles = OrdinalTangles(agreement=4, verbose=False)
    tangles.fit(q.values)
    assert normalized_mutual_info_score(tangles.labels_, data.ys) > 0.99
    hierarchy_truth_list = [[between(0, 10), between(10, 20)], [
        between(20, 30), between(30, 40)]]
    assert aari(BinaryHierarchyTree(hierarchy_truth_list),
                BinaryHierarchyTree(tangles.hierarchy_), 2) == 1.0
