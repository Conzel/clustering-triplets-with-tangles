from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

import numpy as np

from questionnaire import Questionnaire

from cblearn.embedding import SOE


class Baseline():
    def __init__(self, name):
        self.name = name
        if name.lower() == "none" or name is None:
            self.method = Baseline._raise_no_baseline_error
        if name.lower() == "gmm":
            self.method = Baseline._gmm_baseline
        if name.lower() == "soe-gmm":
            self.method = Baseline._soe_gmm_baseline
        if name.lower() == "soe-knn":
            self.method = Baseline._soe_knn_baseline

    def _raise_no_baseline_error(data, questionnaire, n_components):
        raise ValueError("No baseline")

    def _gmm_baseline(xs: np.ndarray, questionnaire: Questionnaire, n_components: int, seed=None):
        """
        Calculates a baseline for the clustering by fitting a gaussian mixture model
        to the data x and inferring labels y'. 

        Returns predicted labels as ndarray.
        """
        gm = GaussianMixture(n_components=n_components,
                             random_state=seed).fit(xs)
        y_pred = gm.predict(xs)
        return y_pred

    def _soe_gmm_baseline(xs: np.ndarray, questionnaire: Questionnaire, n_components: int, seed=None):
        """
        Calculates a baseline for the clustering by fitting a SOE embedding to the data x and
        inferring labels via a GMM (with n_components clusters). 

        Returns predicted labels as ndarray.
        """
        baseline = soe_gmm_baseline(n_components)
        return baseline.fit_predict(*questionnaire.to_bool_array())

    def _soe_knn_baseline(xs: np.ndarray, questionnaire: Questionnaire, n_components: int, seed=None):
        """
        Calculates a baseline for the clustering by fitting a SOE embedding to the data x and
        inferring labels via k-NN (where k = n_components).

        Returns predicted labels as ndarray.
        """
        baseline = soe_knn_baseline(n_components)
        return baseline.fit_predict(*questionnaire.to_bool_array())

    def predict(self, xs: np.ndarray, questionnaire: Questionnaire, n_components: int) -> np.ndarray:
        """
        Uses the chosen baseline to predict the labels of the given data.
        """
        return self.method(xs, questionnaire, n_components)


def soe_gmm_baseline(clusters: int) -> Pipeline:
    """
    Baseline that is gained by first learning a SOE embedding and then
    using the embedding to learn a GMM.

    The baseline can be used like any other sklearn estimator.

    The input must be in (triplets, response) format, as can be learned
    from the cblearn documentation.

    https://cblearn.readthedocs.io/en/latest/references/generated/cblearn.embedding.SOE.html?highlight=soe
    """
    pipe = Pipeline([("soe", SOE(n_components=clusters)),
                    ("gmm", GaussianMixture(n_components=clusters))])
    return pipe


def soe_knn_baseline(clusters: int) -> Pipeline:
    """
    Similar to the soe_gmm baseline but we are using a generic KNN to 
    fit.
    """
    pipe = Pipeline([("soe", SOE(n_components=clusters)),
                     ("knn", KMeans(n_clusters=clusters))])
    return pipe
