from sklearn.metrics import normalized_mutual_info_score
from data_generation import generate_smb_data
from estimators import OrdinalTangles
from questionnaire import Questionnaire
from baselines import soe_knn_baseline
import numpy as np
import random

tangles_scores = []
baseline_scores = []
np.random.seed(0)
random.seed(42)


for _ in range(50):
    graph, ys = generate_smb_data(n=20, k=5, p=0.9, q=0.1)
    q = Questionnaire.from_graph(graph, density=0.1)

    triplets = q.values
    triplets_cblearn = q.to_bool_array()

    tangles = OrdinalTangles(6, verbose=1)
    tangles_labels = tangles.fit_predict(triplets)
    tangles_score = normalized_mutual_info_score(ys, tangles_labels)

    #
    soe_knn = soe_knn_baseline(2, 5)
    baseline_labels = soe_knn.fit_predict(*triplets_cblearn)
    baseline_score = normalized_mutual_info_score(ys, baseline_labels)

    print(f"Tangles score: {tangles_score}\nBaseline score: {baseline_score}")
    baseline_scores.append(baseline_score)
    tangles_scores.append(tangles_score)

print(
    f"Tangles score: {np.mean(tangles_score)}\nBaseline score: {np.mean(baseline_score)}")
