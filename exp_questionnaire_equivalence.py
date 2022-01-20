import numpy as np
from experiment_runner import parameter_variation, Configuration
import yaml
from plotting import AltairPlotter
import sys
import math
from data_generation import generate_gmm_data_fixed_means
from estimators import OrdinalTangles
from questionnaire import generate_questionnaire

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


n = 333
std = 1.0
mean_c = 3.0
seed = 1
density = 0.01

# noise_estim = estimate_noise(0, std, mean_c, std, n, n_runs=100000) * 2
# print(f"This should be equivalent to a noise of {noise_estim}.")

noise_estim = 0.4
data = generate_gmm_data_fixed_means(
    n, np.array([[0.0, 0.0], [0.0, mean_c], [mean_c, 0.0]]), std, seed=None)
# super  strange behaviour... if we set the std here to 0.0,
# the clustering is a lot worse, than if we set it to 0.001 or something like that
data_no_std = generate_gmm_data_fixed_means(
    n, np.array([[0.0, 0.0], [mean_c, 0.0], [0.0, mean_c]]), 0.001, seed=None)

scores = []
scores_no_std = []
for _ in range(10):
    triplets = generate_questionnaire(data.xs, density=density).values
    triplets_no_std = generate_questionnaire(
        data_no_std.xs, noise=noise_estim, density=density, imputation_method="random").values

    tangles = OrdinalTangles(agreement=int(n/3))
    # no noise but std
    tangles.fit(triplets)
    score = tangles.score(triplets, data.ys)
    # print(f"No noise, but std = {std}: {score}")

    # noise but no std
    tangles = OrdinalTangles(agreement=int(n/3))
    tangles.fit(triplets_no_std)
    score_no_std = tangles.score(triplets_no_std, data_no_std.ys)
    # print(f"No std, but noise = {noise_estim}: {score_no_std}")

    scores.append(score)
    scores_no_std.append(score_no_std)

print(f"Mean score: {np.mean(scores)}, std: {np.std(scores)}")
print(
    f"No std, Mean score: {np.mean(scores_no_std)}, std: {np.std(scores_no_std)}")
