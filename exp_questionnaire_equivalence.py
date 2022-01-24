"""
Produces all the plots in the mindset-questionnaire equivalence notes.
https://www.notion.so/Further-Investigations-to-noise-resistance-a361f49116b64d79a93b605b2719adce
"""
import numpy as np
from experiment_runner import parameter_variation, Configuration
import yaml
from plotting import AltairPlotter
import sys
import math
import pandas as pd
import altair as alt
from data_generation import generate_gmm_data_fixed_means
from estimators import OrdinalTangles
from questionnaire import Questionnaire 
from sklearn.metrics import normalized_mutual_info_score

# base_config = Configuration.from_yaml(
#     yaml.load(open("experiments/11-questionnaire-equiv.yaml")))
# # if len(sys.argv) > 1 and (sys.argv[1] == "-p" or sys.argv[1] == "--parallelize"):
# #     workers = None
# # else:
# # workers = 1
# noise = np.arange(0.0, 0.5, 0.05)
# df = parameter_variation(noise, "noise", "noise", base_config, workers=None)
# p = AltairPlotter()
# chart = p.parameter_variation(df, "noise")


def calculate_questionnaire_guarantees(n, k, p, a, m=None):
    """n = points per cluster, k = # clusters, p = noise (solveig)
    See p.12 Klepper et al., Tangles - Algorithmic framework
    """
    N = n * k
    max_noise = 1/(k + 3)
    if m is None:
        m = math.comb(k, 2)
    a_min, a_max = p*N, (1 - 3*p)*N/k
    wrong_clustering_max_perc = k * m * \
        math.exp(-2 * N * (k * a/N - 1 + 3*p)**2/(9 * k))
    spurious_tangle_max_perc = k * m * math.exp(-2 * N * (a/N - p)**2 / k)
    print(f"Unique number of questions: {m}")
    print(f"Max noise: {max_noise}, given noise: {p}")
    print(f"Range for a: ({a_min}, {a_max}), given a: {a}")
    print(f"Wrong clustering max: {wrong_clustering_max_perc}")
    print(f"Spurious tangle max: {spurious_tangle_max_perc}")


def estimate_noise(mean_a, std_a, mean_b, std_b, n_points, n_runs=10000):
    noise_estimate = []
    points_a = np.random.normal(mean_a, std_a, (n_points, 2))
    points_b = np.random.normal([0.0, mean_b], std_b, (n_points, 2))
    for _ in range(n_runs):
        a = points_a[np.random.choice(n_points)]
        b = points_b[np.random.choice(n_points)]
        wrong_a = (np.linalg.norm(a - points_a, axis=1) >
                   np.linalg.norm(b - points_a, axis=1)).sum()
        wrong_b = (np.linalg.norm(a - points_b, axis=1) <
                   np.linalg.norm(b - points_b, axis=1)).sum()
        noise_perc = (wrong_a + wrong_b) / (2 * n_points)
        noise_estimate.append(noise_perc)
    return np.mean(noise_estimate)


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

# We also gotta care for the geometry, if we set the points like this:
# 1  2
# 3
# The the cuts between 2 and 3 will cut through cluster 1 and introduce additional noise.
#
# super  strange behaviour... if we set the std here to 0.0,
# the clustering is a lot worse, than if we set it to 0.0001 or something like that
# data_no_std = generate_gmm_data_fixed_means(
#     n, np.array([[-2*mean_c, mean_c], [-2*mean_c, -mean_c], [2*mean_c, mean_c]]), 0.0001, seed=None)


def noise_mindset_plot(datafunc, density, filter_useless_questions=False, n_runs=10, soft=None):
    scores_no_std = []
    no_clusters = []
    df = pd.DataFrame()
    preds = {}
    data_dict = {}

    # Calculates Mindset example for GMMs
    for noise in np.arange(0.0, 0.51, 0.025):
        # for noise in [0.2]:
        preds[noise] = []
        data_dict[noise] = []
        for _ in range(n_runs):
            data = datafunc()
            triplets_no_std = Questionnaire.from_euclidean(
                data.xs, noise=noise, density=density, imputation_method="random", soft_threshhold=soft, flip_noise=True)
            if filter_useless_questions:
                triplets_no_std = get_useful_cuts(triplets_no_std, 3)
            else:
                triplets_no_std = triplets_no_std.values

            # noise but no std
            tangles = OrdinalTangles(agreement=int(n/3), verbose=True)
            ys_pred = tangles.fit_predict(triplets_no_std)
            score_no_std = normalized_mutual_info_score(
                data.ys, ys_pred)
            clusters = np.unique(ys_pred).size
            print(f"Score: {score_no_std}, #clusters: {clusters}")

            scores_no_std.append(score_no_std)
            no_clusters.append(clusters)
            df = df.append({"nmi": score_no_std, "clusters": clusters,
                            "noise": noise}, ignore_index=True)
            preds[noise].append(ys_pred)
            data_dict[noise].append(data)

        print(
            f"No std, Mean score: {np.mean(scores_no_std)}, std: {np.std(scores_no_std)}")
        print(
            f"No std, Mean #clusters: {np.mean(no_clusters)}, std: {np.std(no_clusters)}")

    df_avg = df.groupby("noise").mean().reset_index()
    circles = alt.Chart(df_avg).mark_circle(color="red").encode(
        x="noise", y="nmi").interactive()
    bars = alt.Chart(df_avg).mark_bar().encode(
        x="noise", y="clusters").interactive()

    chart = alt.layer(bars, circles).resolve_scale(y="independent")
    chart.show()
    return df_avg, data_dict, preds, chart


def make_datafunc(seed):
    return lambda: generate_gmm_data_fixed_means(n, np.array(
        [[-2*mean_c, mean_c], [-2*mean_c, -mean_c], [2*mean_c, mean_c]]), 1.0, seed=seed)


def make_datafunc_no_std(seed):
    return lambda: generate_gmm_data_fixed_means(n, np.array(
        [[-2*mean_c, mean_c], [-2*mean_c, -mean_c], [2*mean_c, mean_c]]), 0.001, seed=seed)


# no std setup
df_no_std, data_dict_normal_no_std, preds_normal_no_std, chart_normal_no_std = noise_mindset_plot(
    make_datafunc(None), density)
chart_normal_no_std.show()

# # normal setup
df, data_dict_normal, preds_normal, chart_normal = noise_mindset_plot(
    make_datafunc(None), density)
p = AltairPlotter()
d1 = data_dict_normal[0.2][2]
ypred1 = preds_normal[0.2][2]
# showing the ground truth as example
c1 = p.assignments(d1.xs, d1.ys)
c2 = p.assignments(d1.xs, ypred1)
c1.show()
c2.show()
chart_normal.show()

# with only useful questions
df_useful, data_dict_useful, preds_useful, chart_useful = noise_mindset_plot(
    make_datafunc(None), density, filter_useless_questions=True)
chart_useful.show()

df_soft, data_dict_soft, preds_soft, chart_soft = noise_mindset_plot(
    make_datafunc(None), density, filter_useless_questions=False, soft=8.0)
chart_soft.show()


def make_datafunc_bad_geometry(seed):
    # points are set such that the cuts between 2 and 3 will cut through cluster 1
    # and introduce additional noise.
    return lambda: generate_gmm_data_fixed_means(n, np.array(
        [[-2*mean_c, mean_c], [-2*mean_c, -mean_c], [2*mean_c, 0]]), 1.0, seed=seed)


# # with bad geometry
df_bad, data_dict_bad, preds_bad, chart_bad = noise_mindset_plot(
    make_datafunc_bad_geometry(None), density, filter_useless_questions=True)
chart_bad.show()
