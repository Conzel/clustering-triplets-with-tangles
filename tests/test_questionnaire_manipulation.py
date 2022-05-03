from data_generation import generate_gmm_data_fixed_means
from questionnaire import Questionnaire
import numpy as np
import pytest


def test_fill_with():
    data = generate_gmm_data_fixed_means(n=3, means=np.array(
        [[-6, 3], [-6, -3], [6, 3]]), std=1, seed=0)
    q_miss = Questionnaire.from_metric(data.xs, noise=0.9)
    q_fill = Questionnaire.from_metric(data.xs)
    q_filled = q_miss.fill_with(q_fill)
    assert np.all(q_filled.values == q_fill.values)
    assert np.all(q_filled.labels == q_fill.labels)


def test_order_labels():
    q = Questionnaire(np.array([[0, 1], [-1, -1]]), [(0, 1), (2, 1)])
    assert not q.labels_are_ordered()
    q = q.order_labels()
    assert q.labels_are_ordered()
    assert np.all(q.values == np.array([[0, 0], [-1, -1]]))
    assert np.all(q.labels == [(0, 1), (1, 2)])


def test_sort_labels_raises_unordered():
    q = Questionnaire(np.array([[0, 1], [-1, -1]]), [(2, 0), (1, 2)])
    with pytest.raises(ValueError):
        q.sort_labels()


def test_sort_labels():
    q = Questionnaire(np.array(
        [[0, 1], [-1, -1], [1, 0], [-1, 0]]).T, [(1, 2), (0, 2), (0, 1), (3, 4)])
    q = q.sort_labels()
    labels_ = [(0, 1), (0, 2), (1, 2), (3, 4)]
    vals_ = np.array([[1, 0], [-1, -1], [0, 1], [-1, 0]]).T
    assert q.labels == labels_
    assert np.all(q.values == vals_)

def test_equivalence():
    q1 = Questionnaire(np.array([[0, 1], [-1, 1]]), [(2, 1), (2, 0)])
    q2 = Questionnaire(np.array([[0, 1], [0, -1]]), [(0,2), (1,2)])
    assert q1.equivalent(q2)

def test_normal_form():
    # Order: flip all non-missing entries
    # Sort: swap columns according to sorting
    # (2,1)  (2,0)            (0, 2) (1, 2)
    # 0        1     =====>     0      1
    # -1       1                0     -1
    q = Questionnaire(np.array([[0, 1], [-1, 1]]), [(2, 1), (2, 0)])
    assert np.all(q.normal_form().values == np.array([[0, 1], [0, -1]]))
    assert np.all(q.normal_form().labels == np.array([(0, 2), (1, 2)]))
