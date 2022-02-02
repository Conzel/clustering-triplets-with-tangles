#!/usr/bin/env python3
"""
Script for testing tangles algorithm performance on real world datasets.
"""
import sklearn.datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from baselines import soe_knn_baseline
from estimators import OrdinalTangles
from plotting import AltairPlotter
from questionnaire import Questionnaire

# Setting up data and transformers
wine = sklearn.datasets.load_wine()
standardized_data = StandardScaler().fit_transform(wine.data)
tangles = OrdinalTangles(agreement=7, verbose=False)
baseline = soe_knn_baseline(clusters=3)

questionnaire = Questionnaire.from_euclidean(standardized_data)
ys_baseline = baseline.fit_predict(*questionnaire.to_bool_array())
ys_tangles = tangles.fit_predict(questionnaire.values)

tsne = TSNE(n_components=2)
embedding = tsne.fit_transform(standardized_data)

p = AltairPlotter()

c = p.comparison(embedding, wine.target, ys_tangles) & p.comparison(
    embedding, wine.target, ys_baseline)
c.show()
