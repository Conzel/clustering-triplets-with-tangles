"""
Produces all the plots in the mindset-questionnaire equivalence notes.
https://www.notion.so/Further-Investigations-to-noise-resistance-a361f49116b64d79a93b605b2719adce
"""
from baselines import Baseline
from sklearn.metrics import normalized_mutual_info_score
from questionnaire import Questionnaire
from estimators import OrdinalTangles
from data_generation import generate_gmm_data_fixed_means
import altair as alt
import pandas as pd
from plotting import AltairPlotter
import numpy as np
import sys
sys.path.append("..")


def get_useful_cuts(q: Questionnaire, k):
    assignments = np.mod(q.labels, k)
    filter_rel = assignments[:, 0] != assignments[:, 1]
    return q.values[:, filter_rel]


def get_useless_cuts(q: Questionnaire, k):
    assignments = np.mod(q.labels, k)
    filter_rel = assignments[:, 0] == assignments[:, 1]
    return q.values[:, filter_rel]


n = 333
std = 1.0
mean_c = 3.0
density = 0.001
seed = 1

class MindsetResult():
    def __init__(self, df_avg, data_dict, preds, chart, preds_baseline, embedding):
        self.df_avg = df_avg
        self.data_dict = data_dict
        self.preds = preds
        self.chart = chart
        self.preds_baseline = preds_baseline
        self.embedding = embedding

        

def noise_mindset_plot(datafunc, density, filter_useless_questions=False, n_runs=10, soft=None, baseline_name=None):
    scores_tangles = []
    no_clusters = []
    df = pd.DataFrame()
    preds = {}
    preds_baseline = {}
    data_dict = {}
    embeddings = {}

    # Calculates Mindset example for GMMs
    for noise in np.arange(0.0, 0.51, 0.025):
        # for noise in [0.2]:
        preds[noise] = []
        data_dict[noise] = []
        preds_baseline[noise] = []
        embeddings[noise] = []
        for _ in range(n_runs):
            data = datafunc()
            questionnaire = Questionnaire.from_euclidean(
                data.xs, noise=noise, density=density, imputation_method="random", soft_threshhold=soft, flip_noise=True)
            if filter_useless_questions:
                triplets = get_useful_cuts(questionnaire, 3)
            else:
                triplets = questionnaire.values

            if baseline_name is not None:
                baseline = Baseline(baseline_name)
                ys_base_pred = baseline.predict(data.xs, questionnaire, 3)
                score_baseline = normalized_mutual_info_score(
                    data.ys, ys_base_pred)
                preds_baseline[noise].append(ys_base_pred)
                embeddings[noise].append(baseline._embedding)

            # noise but no std
            tangles = OrdinalTangles(agreement=int(n/3), verbose=True)
            ys_pred = tangles.fit_predict(triplets)
            score = normalized_mutual_info_score(
                data.ys, ys_pred)
            clusters = np.unique(ys_pred).size
            print(f"Score: {score}, #clusters: {clusters}")

            scores_tangles.append(score)
            no_clusters.append(clusters)
            if baseline_name is not None:
                df = df.append({"nmi": score, "clusters": clusters,
                                "noise": noise, "nmi_baseline": score_baseline}, ignore_index=True)
            else:
                df = df.append({"nmi": score, "clusters": clusters,
                                "noise": noise}, ignore_index=True)
            preds[noise].append(ys_pred)
            data_dict[noise].append(data)

        print(
            f"Mean score: {np.mean(scores_tangles)}, std: {np.std(scores_tangles)}")
        print(
            f"Mean #clusters: {np.mean(no_clusters)}, std: {np.std(no_clusters)}")

    df_avg = df.groupby("noise").mean().reset_index()
    circles = alt.Chart(df_avg).mark_circle(color="red").encode(
        x="noise", y="nmi").interactive()
    bars = alt.Chart(df_avg).mark_bar().encode(
        x="noise", y="clusters").interactive()
    if baseline_name is not None:
        baseline_markers = alt.Chart(df_avg).mark_line(color="green").encode(
            x="noise", y="nmi_baseline").interactive()
        chart = alt.layer(bars, circles, baseline_markers).resolve_scale(
            y="independent")
    else:
        chart = alt.layer(bars, circles).resolve_scale(y="independent")
    return MindsetResult(df_avg, data_dict, preds, chart, preds_baseline, embeddings)


def make_datafunc(seed):
    return lambda: generate_gmm_data_fixed_means(n, np.array(
        [[-2*mean_c, mean_c], [-2*mean_c, -mean_c], [2*mean_c, mean_c]]), 1.0, seed=seed)


def make_datafunc_no_std(seed):
    return lambda: generate_gmm_data_fixed_means(n, np.array(
        [[-2*mean_c, mean_c], [-2*mean_c, -mean_c], [2*mean_c, mean_c]]), 0.001, seed=seed)


# no std setup
no_std_result = noise_mindset_plot(
    make_datafunc(None), density)
no_std_result.chart.show()

# # normal setup
normal_result = noise_mindset_plot(
    make_datafunc(None), density)
p = AltairPlotter()
d1 = normal_result.data_dict[0.2][2]
ypred1 = normal_result.preds[0.2][2]
# showing the ground truth as example
c1 = p.assignments(d1.xs, d1.ys)
c2 = p.assignments(d1.xs, ypred1)
c1.show()
c2.show()
normal_result.chart.show()

# with only useful questions
result_useful = noise_mindset_plot(
    make_datafunc(None), density, filter_useless_questions=True)
result_useful.chart.show()

result_soft = noise_mindset_plot(
    make_datafunc(None), density, filter_useless_questions=False, soft=8.0)
result_soft.chart.show()


def make_datafunc_bad_geometry(seed):
    # points are set such that the cuts between 2 and 3 will cut through cluster 1
    # and introduce additional noise.
    return lambda: generate_gmm_data_fixed_means(n, np.array(
        [[-2*mean_c, mean_c], [-2*mean_c, -mean_c], [2*mean_c, 0]]), 1.0, seed=seed)


# with bad geometry
# Points are placed as such:
# 1
#       3
# 2
# This causes cuts between 1 and 2 to often go through 3
result_bad = noise_mindset_plot(
    make_datafunc_bad_geometry(None), density, filter_useless_questions=True, n_runs=10, baseline_name="soe-kmeans")
result_bad.chart.show()

result_bad_silhouette = noise_mindset_plot(
    make_datafunc_bad_geometry(None), density, filter_useless_questions=True, n_runs=1, baseline_name="soe-kmeans-silhouette")
result_bad_silhouette.chart.show()

