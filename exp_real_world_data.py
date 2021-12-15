#!/usr/bin/env python3
"""
Script for testing tangles algorithm performance on real world datasets.
"""

import sklearn.datasets
from experiment_runner import tangles_hard_predict
from questionnaire import generate_questionnaire
from sklearn.metrics import normalized_mutual_info_score

wine = sklearn.datasets.load_wine()

questionnaire = generate_questionnaire(wine.data).values
ys_tangles = tangles_hard_predict(questionnaire, agreement=20)

print(normalized_mutual_info_score(wine.target, ys_tangles))
