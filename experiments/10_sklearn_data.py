#!/usr/bin/env python3
"""
Script for testing tangles algorithm performance on real world datasets.
"""
import sklearn.datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

from baselines import soe_kmeans_baseline
from estimators import OrdinalTangles
from plotting import AltairPlotter
from questionnaire import Questionnaire

# Setting up data and transformers
wine = sklearn.datasets.load_wine()
standardized_data = StandardScaler().fit_transform(wine.data)

questionnaire = Questionnaire.from_metric(standardized_data, density=0.1)

tangles = OrdinalTangles(agreement=12, verbose=False)
ys_tangles = tangles.fit_predict(questionnaire.values)
nmi_tangles = normalized_mutual_info_score(ys_tangles, wine.target)

baseline = soe_kmeans_baseline(clusters=3, data_dimension=2)
ys_baseline = baseline.fit_predict(*questionnaire.to_bool_array())
nmi_baseline = normalized_mutual_info_score(ys_baseline, wine.target)

tsne = TSNE(n_components=2)
embedding = tsne.fit_transform(standardized_data)

p = AltairPlotter()

c = p.comparison(embedding, wine.target, ys_tangles).properties(title=f"Wine dataset: Tangles (NMI = {nmi_tangles:.2f})") & p.comparison(
    embedding, wine.target, ys_baseline).properties(title=f"Wine dataset: SOE-kMeans (NMI = {nmi_baseline:.2f})")
c.show()
