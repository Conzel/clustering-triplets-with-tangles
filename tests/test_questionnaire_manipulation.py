from data_generation import generate_gmm_data_fixed_means
from questionnaire import Questionnaire
import numpy as np


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
