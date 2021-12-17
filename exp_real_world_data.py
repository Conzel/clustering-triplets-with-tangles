#!/usr/bin/env python3
"""
Script for testing tangles algorithm performance on real world datasets.
"""

from matplotlib.pyplot import stackplot
import sklearn.datasets
from experiment_runner import tangles_hard_predict
from questionnaire import generate_questionnaire
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

wine = sklearn.datasets.load_wine()
standardized_data = StandardScaler().fit_transform(wine.data)

questionnaire = generate_questionnaire(standardized_data).values
ys_tangles = tangles_hard_predict(questionnaire, agreement=7)

print(normalized_mutual_info_score(wine.target, ys_tangles))
