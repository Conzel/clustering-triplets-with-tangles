import unittest
from sklearn.datasets import load_iris
from data_generation import generate_gmm_data_fixed_means
from estimators import OrdinalTangles
from experiment_runner import Questionnaire
import numpy as np


class OrdinalTanglesTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_predict_subset_equal(self):
        iris = load_iris()
        tangles = OrdinalTangles(agreement=5, verbose=False)
        q = Questionnaire.from_metric(iris.data)

        tangles.fit(q.values)
        ys1 = tangles.predict(q.values[:10])
        ys2 = tangles.predict(q.values)
        self.assertTrue((ys1 == ys2[:10]).all())

    def test_estimator_performance(self):
        synthetic_data = generate_gmm_data_fixed_means(
            n=15, means=np.array(np.array([[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]])), std=0.5, seed=1)
        tangles = OrdinalTangles(agreement=5, verbose=False)
        q = Questionnaire.from_metric(synthetic_data.xs)
        tangles.fit(q.values)
        self.assertAlmostEqual(tangles.score(q.values, synthetic_data.ys), 1.0)

    def tearDown(self):
        pass
