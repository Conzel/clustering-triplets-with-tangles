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
from sklearn.manifold import TSNE
import altair as alt
import pandas as pd

wine = sklearn.datasets.load_wine()
standardized_data = StandardScaler().fit_transform(wine.data)

questionnaire = generate_questionnaire(standardized_data).values
ys_tangles = tangles_hard_predict(questionnaire, agreement=7)

print(normalized_mutual_info_score(wine.target, ys_tangles))

tsne = TSNE(n_components=2)
tsne_data = tsne.fit_transform(standardized_data)

# wine_data = pd.DataFrame(wine.data, columns=wine.feature_names)
tsne_data = pd.DataFrame(tsne_data, columns=["x", "y"])
tsne_data["label"] = wine.target
tsne_data["prediction_tangles"] = ys_tangles
tsne_data["correct"] = tsne_data["label"] == tsne_data["prediction_tangles"]
chart_true = alt.Chart(tsne_data).mark_point().encode(
    x="x",
    y="y",
    color="label:N"
).properties(
    title="True labels"
)
chart_pred = chart_true.encode(
    color="prediction_tangles:N").properties(title="Predicted labels")
chart_missed = chart_true.encode(
    color="correct:N").properties(title="Correctly classified")

chart_combined = (chart_true | chart_pred) & chart_missed
chart_combined.show()
