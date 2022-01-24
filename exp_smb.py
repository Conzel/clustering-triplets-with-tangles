from sklearn.metrics import normalized_mutual_info_score
from data_generation import generate_smb_data
from estimators import OrdinalTangles
from questionnaire import Questionnaire
from baselines import soe_knn_baseline
import numpy as np
import random
import pandas as pd
import altair as alt

np.random.seed(0)
random.seed(42)


def smb_performance(p, q, n=20, k=5, n_runs=10):
    tangles_scores = []
    baseline_scores = []
    for _ in range(n_runs):
        graph, ys = generate_smb_data(n=n, k=k, p=p, q=q)
        questionnaire = Questionnaire.from_graph(graph, density=0.1)

        triplets = questionnaire.values
        triplets_cblearn = questionnaire.to_bool_array()

        tangles = OrdinalTangles(6, verbose=1)
        tangles_labels = tangles.fit_predict(triplets)
        tangles_score = normalized_mutual_info_score(ys, tangles_labels)

        #
        soe_knn = soe_knn_baseline(2, 5)
        baseline_labels = soe_knn.fit_predict(*triplets_cblearn)
        baseline_score = normalized_mutual_info_score(ys, baseline_labels)

        baseline_scores.append(baseline_score)
        tangles_scores.append(tangles_score)

    return np.mean(tangles_scores), np.mean(baseline_scores)


# simple example
tangles_nmi, baseline_nmi = smb_performance(0.9, 0.1)
print(f"Tangles: {tangles_nmi}\nBaseline: {baseline_nmi}")

# meshgrid
p, q = np.meshgrid(np.arange(0.1, 1.0, 0.1), np.concatenate(
    ([0.01], np.arange(0.05, 0.5, 0.05))))

tangles_nmi_grid = np.zeros_like(p)
baseline_nmi_grid = np.zeros_like(q)
for i in range(p.shape[0]):
    for j in range(p.shape[1]):
        nmi_score, baseline_score = smb_performance(p[i, j], q[i, j])
        tangles_nmi_grid[i, j] = nmi_score
        baseline_nmi_grid[i, j] = baseline_score

tangles_nmi_df = pd.DataFrame(
    {'p': p.ravel(), 'q': q.ravel(), 'nmi': tangles_nmi_grid.ravel()})
baseline_nmi_df = pd.DataFrame(
    {'p': p.ravel(), 'q': q.ravel(), 'nmi': baseline_nmi_grid.ravel()})

tangles_chart = alt.Chart(tangles_nmi_df).mark_rect().encode(
    x=alt.X('p:O', axis=alt.Axis(title='p', format=".2")),
    y=alt.Y('q:O', axis=alt.Axis(title='q', format=".2"),
            sort=alt.EncodingSortField('q', order='descending'),
            ),
    color='nmi:Q'
).properties(title="Tangles").interactive()

baseline_chart = alt.Chart(baseline_nmi_df).mark_rect().encode(
    x=alt.X('p:O', axis=alt.Axis(title='p', format=".2")),
    y=alt.Y('q:O', axis=alt.Axis(title='q', format=".2"),
            sort=alt.EncodingSortField('q', order='descending'),
            ),
    color='nmi:Q'
).properties(title="Baseline").interactive()

(baseline_chart | tangles_chart).show()
