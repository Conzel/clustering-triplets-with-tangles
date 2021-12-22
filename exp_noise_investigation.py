import os
from questionnaire import generate_questionnaire
from experiment_runner import Configuration, _generate_data, ImputationMethod, Baseline
from tangles.data_types import Cuts, Data
from tangles.utils import normalize, compute_hard_predictions
from tangles.tree_tangles import ContractedTangleTree, compute_soft_predictions_children, tangle_computation
from tangles.cost_functions import mean_manhattan_distance
from tangles.plotting import labels_to_colors, plot_dataset
import random
import yaml
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from functools import partial
from pathlib import Path


# --- setup
with open("experiments/09-noise-investigation.yaml", "r") as f:
    conf = Configuration.from_yaml(yaml.safe_load(f))
base_folder = Path("results/09-noise-investigation")
save_figs = True
os.makedirs(base_folder, exist_ok=True)
data = _generate_data(conf)

verbose = True

np.random.seed(conf.seed)
random.seed(conf.seed)

questionnaire = generate_questionnaire(
    data.xs, noise=conf.noise, density=conf.density, seed=conf.seed, imputation_method=ImputationMethod(
        conf.imputation_method),
    verbose=verbose).values

questionnaire_exact = generate_questionnaire(
    data.xs, noise=0.0, density=conf.density, seed=conf.seed, imputation_method=ImputationMethod(
        conf.imputation_method),
    verbose=verbose).values


def compute_cost_and_order_cuts2(bipartitions, cost_function, verbose=True):
    if verbose:
        print("Computing costs of cuts...")

    bipartitions = deepcopy(bipartitions)

    cost_bipartitions = np.zeros(len(bipartitions.values), dtype=float)
    for i_cut, cut in enumerate(tqdm(bipartitions.values, disable=not verbose)):
        cost_bipartitions[i_cut] = cost_function(cut)

    idx = np.argsort(cost_bipartitions)

    bipartitions.values = bipartitions.values[idx]
    bipartitions.costs = cost_bipartitions[idx]
    if bipartitions.names is not None:
        bipartitions.names = bipartitions.names[idx]
    if bipartitions.equations is not None:
        bipartitions.equations = bipartitions.equations[idx]

    bipartitions.order = np.argsort(idx)

    return bipartitions, idx


# Interpreting the questionnaires as cuts and computing their costs
bipartitions = Cuts((questionnaire == 1).T)
cost_fn = partial(mean_manhattan_distance,
                  questionnaire, conf.num_distance_function_samples)
cuts, idx = compute_cost_and_order_cuts2(bipartitions, cost_fn)

bipartitions_exact = Cuts((questionnaire_exact == 1).T)

# Building the tree, contracting and calculating predictions
tangles_tree = tangle_computation(cuts=cuts,
                                  agreement=conf.agreement,
                                  # print nothing
                                  verbose=int(verbose)
                                  )

contracted = ContractedTangleTree(tangles_tree)
contracted.prune(5, verbose=verbose)

contracted.calculate_setP()

# soft predictions
weight = np.exp(-normalize(cuts.costs))
compute_soft_predictions_children(
    node=contracted.root, cuts=cuts, weight=weight, verbose=3)

ys_predicted, _ = compute_hard_predictions(
    contracted, cuts=cuts, verbose=verbose)

baseline = Baseline("gmm")
y_base = baseline.predict(data, 5)
print(normalized_mutual_info_score(data.ys, y_base))

# --- plotting
cmap_groundtruth = plt.cm.get_cmap('autumn')
cmap_predictions = plt.cm.get_cmap('cool')

if data.ys is not None:
    fig, (ax_true, ax_predicted) = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 5))
    colors_true = labels_to_colors(data.ys, cmap=cmap_groundtruth)
    ax_true.set_title("Ground truth clusters")
    _ = plot_dataset(
        data, colors_true, ax=ax_true, add_colorbar=False)
else:
    fig, ax_predicted = plt.subplots(nrows=1, ncols=1, figsize=(10, 50))

colors_predicted = labels_to_colors(
    ys_predicted, cmap=cmap_predictions)
_ = plot_dataset(data, colors_predicted,
                 ax=ax_predicted, add_colorbar=False)
ax_predicted.set_title("Predicted clusters")

plt.tight_layout()
if save_figs:
    plt.savefig(base_folder / f"hard_clustering.svg")
else:
    plt.show()

# --- Soft predictions
# plotting.plot_soft_predictions(data, contracted)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_cut(data: Data, bipartition: np.ndarray, correct_labels=None) -> None:
    """
    bipartition: np.ndarray
    List of truth values, true if the corresponding datapoint is in the
    left partition, false otherwise.
    """
    assert data.xs.shape[0] == len(bipartition)
    assert data.xs.shape[1] == 2

    cm = plt.get_cmap("tab10")
    c1 = cm(0)
    c2 = cm(1)
    c3 = cm(3)
    c4 = cm(4)

    plt.figure()
    # cheapest cut
    if correct_labels is None:
        plt.plot(data.xs[:, 0][bipartition], data.xs[:, 1]
                 [bipartition], '.', markersize=12, color=c1)
        plt.plot(data.xs[:, 0][np.logical_not(bipartition)], data.xs[:, 1]
                 [np.logical_not(bipartition)], '.', markersize=12, color=c2)
    else:
        mask_both_true = np.logical_and(bipartition, correct_labels)
        mask_both_false = np.logical_and(np.logical_not(
            bipartition), np.logical_not(correct_labels))
        mask_false_correct_label_true = np.logical_and(
            np.logical_not(bipartition), correct_labels)
        mask_true_correct_label_false = np.logical_and(
            bipartition, np.logical_not(correct_labels))
        masks = [mask_both_true, mask_both_false,
                 mask_true_correct_label_false, mask_false_correct_label_true]
        colors = [c1, c2, c3, c4]
        labels = ["both true", "both false",
                  "true correct label false", "false correct label true", ]
        for mask, c, l in zip(masks, colors, labels):
            plt.plot(data.xs[:, 0][mask], data.xs[:, 1][mask], '.', color=c,
                     markersize=12, label=l)
        plt.legend()
    plt.set_cmap("tab10")


def estimate_noise(noisy_cut, exact_cut):
    return 1 - (noisy_cut == exact_cut).sum() / noisy_cut.shape[0]


def num_suffix(i):
    if i == 0:
        raise ValueError("i must be non-zero")
    if i <= 3:
        return ["st", "nd", "rd"][i - 1]
    else:
        return "th"


i = 0
j = 0
threshhold = 0.0
# plot cheapest cuts that are above threshhold
while j < 5 and i < cuts.values.shape[0]:
    cut_normal = cuts.values[i]
    cut_exact = bipartitions_exact.values[idx[i]]
    noise_estim = estimate_noise(cut_normal, cut_exact)
    i += 1
    if noise_estim < threshhold:
        continue

    j += 1
    plot_cut(data, cut_normal, correct_labels=cut_exact)
    plt.title(f"{i}-{num_suffix(i)} cheapest cut: {noise_estim:.2f}")
    if save_figs:
        plt.savefig(base_folder / f"hard_clustering_cut_{i}.svg")
    else:
        plt.show()

# plot the splitting cuts
for i in [2, 13, 19, 26, 209, 377]:
    cut_normal = cuts.values[i]
    cut_exact = bipartitions_exact.values[idx[i]]
    noise_estim = estimate_noise(cut_normal, cut_exact)

    plot_cut(data, cut_normal, correct_labels=cut_exact)
    plt.title(f"{i}-{num_suffix(i)} cheapest cut: {noise_estim:.2f}")
    if save_figs:
        plt.savefig(base_folder / f"splitting_cut_{i}.svg")
    else:
        plt.show()

total_noise = []
for i in range(cuts.values.shape[0]):
    cut_normal = cuts.values[i]
    cut_exact = bipartitions_exact.values[idx[i]]
    total_noise.append(estimate_noise(cut_normal, cut_exact))

print(
    f"Total noise estimation: {sum(total_noise)/cuts.values.shape[0]:.2f}")

plt.figure()
plt.hist(total_noise)
plt.title("Histogram of noise of cuts")
if save_figs:
    plt.savefig(base_folder / f"hist_noise.svg")
else:
    plt.show()

# We see that cuts with more noise are more likely to be used in early stages,
# e.g. have low cost.
plt.figure()
plt.scatter(range(len(total_noise)), total_noise)
plt.title("Noise vs cut order")
if save_figs:
    plt.savefig(base_folder / f"noise_vs_order.svg")
else:
    plt.show()
# fig = px.scatter(x=range(len(total_noise)), y=total_noise)
# fig.show()
