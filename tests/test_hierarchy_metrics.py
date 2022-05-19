from comparison_hc import ComparisonHC
from hierarchies import HierarchyList, aari
from data_generation import generate_gmm_data_fixed_means, generate_planted_hierarchy
from questionnaire import Questionnaire
import numpy as np


def get_level_2_hierarchy():
    return [[[0, 2], [1, 3]], [[4, 7], [6, 5]]]


def get_level_1_hierarchy():
    return [[0, 1], [2, 3]]


def test_aari():
    assert aari(HierarchyList(get_level_2_hierarchy()),
                HierarchyList(get_level_2_hierarchy())) == 1.0
    assert aari(HierarchyList(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), HierarchyList(get_level_2_hierarchy())) == 0.4166666666666667  # regression test


def between(a, b):
    return list(range(a, b))


def test_aari_comparison_hc():
    data = generate_planted_hierarchy(2, 10, 3, 1.0)
    t, r = Questionnaire.from_precomputed(
        data.xs, density=0.1).to_bool_array()
    chc = ComparisonHC(4)
    chc.fit_predict(t, r)
    hierarchy_truth_list = [[between(0, 10), between(10, 20)], [
        between(20, 30), between(30, 40)]]
    assert chc.aari(HierarchyList(hierarchy_truth_list)) == 1.0
