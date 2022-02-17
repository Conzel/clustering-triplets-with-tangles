import numpy as np
from cblearn.embedding import SOE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

from questionnaire import Questionnaire


class Baseline():
    def __init__(self, name):
        self.name = name
        if name.lower() == "none" or name is None:
            self.method = Baseline._raise_no_baseline_error
        if name.lower() == "gmm":
            self.method = Baseline._gmm_baseline
        if name.lower() == "soe-gmm":
            self.method = Baseline._soe_gmm_baseline
        if name.lower() == "soe-kmeans":
            self.method = Baseline._soe_kmeans_baseline
        if name.lower() == "soe-kmeans-silhouette":
            self.method = Baseline._soe_kmeans_silhouette_baseline
        self._embedding = None

    def _raise_no_baseline_error(self, data, questionnaire, n_components):
        raise ValueError("No baseline")

    def _gmm_baseline(self, xs: np.ndarray, questionnaire: Questionnaire, n_components: int, seed=None):
        """
        Calculates a baseline for the clustering by fitting a gaussian mixture model
        to the data x and inferring labels y'. 

        Returns predicted labels as ndarray.
        """
        gm = GaussianMixture(n_components=n_components,
                             random_state=seed).fit(xs)
        y_pred = gm.predict(xs)
        return y_pred

    def _soe_kmeans_silhouette_baseline(self, xs: np.ndarray, questionnaire: Questionnaire, seed: int = None, k_max: int = 20) -> np.ndarray:
        """
        Similar to soe_kmeans, but we are using the silhouette beforehand to best 
        determine the k for k-means.

        Refer to 'find_k_silhouette' for more info on the silhouette method.
        """
        data_dimension = xs.shape[1]
        soe = SOE(n_components=data_dimension, random_state=seed)
        embedding = soe.fit_transform(*questionnaire.to_bool_array())
        self._embedding = embedding
        optimal_kmeans = KMeans(find_k_silhouette(embedding, k_max))
        return optimal_kmeans.fit_predict(embedding)

    def _soe_gmm_baseline(self, xs: np.ndarray, questionnaire: Questionnaire, n_components: int, seed: int = None) -> np.ndarray:
        """
        Calculates a baseline for the clustering by fitting a SOE embedding to the data x and
        inferring labels via a GMM (with n_components clusters). 

        Returns predicted labels as ndarray.
        """
        data_dimension = xs.shape[1]
        baseline = soe_gmm_baseline(data_dimension, n_components)
        return baseline.fit_predict(*questionnaire.to_bool_array())

    def _soe_kmeans_baseline(self, xs: np.ndarray, questionnaire: Questionnaire, n_components: int, seed: int = None) -> np.ndarray:
        """
        Calculates a baseline for the clustering by fitting a SOE embedding to the data x and
        inferring labels via k-NN (where k = n_components).

        Returns predicted labels as ndarray.
        """
        data_dimension = xs.shape[1]
        soe = SOE(n_components=data_dimension, random_state=seed)
        embedding = soe.fit_transform(*questionnaire.to_bool_array())
        kmeans = KMeans(n_clusters=n_components)
        self._embedding = embedding
        return kmeans.fit_predict(embedding)

    def predict(self, xs: np.ndarray, questionnaire: Questionnaire, n_components: int) -> np.ndarray:
        """
        Uses the chosen baseline to predict the labels of the given data.
        """
        return self.method(self, xs, questionnaire, n_components)


def find_k_silhouette(xs: np.ndarray, k_max: int = 20) -> int:
    """
        When using, keep in mind that k_max of 20 might not be necessarily desirable

        We use the Silhouette method of finding an optimal k as a starter,
        since it's pretty easy. Different methods of finding optimal k might be gleaned
        from von Luxburg: https://arxiv.org/abs/1007.1075

        Blog article where the silhouette method is described:
        https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
    """
    sil = []
    ks = list(range(2, k_max + 1))

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in ks:
        kmeans = KMeans(n_clusters=k).fit(xs)
        labels = kmeans.labels_
        sil.append(silhouette_score(xs, labels, metric='euclidean'))

    optimal_k = ks[np.argmax(sil)]
    return optimal_k


def soe_gmm_baseline(data_dimension: int, clusters: int) -> Pipeline:
    """
    Baseline that is gained by first learning a SOE embedding and then
    using the embedding to learn a GMM.

    The baseline can be used like any other sklearn estimator.

    The input must be in (triplets, response) format, as can be learned
    from the cblearn documentation.

    https://cblearn.readthedocs.io/en/latest/references/generated/cblearn.embedding.SOE.html?highlight=soe
    """
    pipe = Pipeline([("soe", SOE(n_components=data_dimension)),
                    ("gmm", GaussianMixture(n_components=clusters))])
    return pipe


def soe_kmeans_baseline(data_dimension: int, clusters: int, seed: int = None) -> Pipeline:
    """
    Similar to the soe_gmm baseline but we are using a generic KNN to 
    fit.
    """
    pipe = Pipeline([("soe", SOE(n_components=data_dimension, random_state=seed)),
                     ("knn", KMeans(n_clusters=clusters, random_state=seed))])
    return pipe
