import unittest
from sklearn.datasets import load_iris
from estimators import OrdinalTangles
from experiment_runner import generate_questionnaire


class OrdinalTanglesTestCase(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        self.data = iris.data
        self.labels = iris.target

    def test_ordinal_tangles(self):
        tangles = OrdinalTangles(agreement=5, verbose=False)
        q = generate_questionnaire(self.data)

        tangles.fit(q.values)
        ys1 = tangles.predict(q.values[:10])
        ys2 = tangles.predict(q.values)
        self.assertTrue((ys1 == ys2[:10]).all())

    def tearDown(self):
        pass
