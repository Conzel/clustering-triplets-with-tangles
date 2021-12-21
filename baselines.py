from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from cblearn.datasets import make_random_triplets
from cblearn.embedding import SOE
from cblearn.metrics import QueryScorer
from experiment_runner import tangles_hard_predict
from plotting import AltairPlotter

from questionnaire import Questionnaire, generate_questionnaire

# Davids example
iris = load_iris()
X = iris.data
q = generate_questionnaire(X)
triplets, responses = q.to_bool_array()
# triplets, responses = make_random_triplets(X, "list-boolean", size=10000)
# Do note: The make random triplets method might produce duplicates.
# Example:
# import pandas as pd
# df = pd.DataFrame(triplets)
# df.duplicated().sum()


estimator = SOE(n_components=2)
scores = cross_val_score(estimator, triplets, responses,
                         scoring=QueryScorer, cv=5)
print(f"The 5-fold CV triplet error is {sum(scores) / len(scores)}.")

embedding = estimator.fit_transform(triplets, responses)
print(f"The embedding has shape {embedding.shape}.")


# kmeans
kmeans = KMeans(n_clusters=3).fit(X)

p = AltairPlotter()
p.comparison(embedding, iris.target, kmeans.labels_)
p.show()

# ys_tangles = tangles_hard_predict(q.values, agreement=4)
# p.comparison(embedding, iris.target, ys_tangles)
# p.show()

