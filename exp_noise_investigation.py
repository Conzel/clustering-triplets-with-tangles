#!/usr/bin/env python3
"""
Investigates how noise and agreement parameter in the tangles algorithm 
correlate and how well the tangles performance is under this case.
"""
import os
import random
from copy import deepcopy
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

from experiment_runner import Configuration, _generate_data
from baselines import Baseline
from questionnaire import Questionnaire, ImputationMethod
from tangles.data_types import Cuts, Data
from tangles.plotting import labels_to_colors, plot_dataset
from tangles.tree_tangles import (ContractedTangleTree,
                                  compute_soft_predictions_children,
                                  tangle_computation)
from tangles.utils import compute_hard_predictions, normalize

from tangles.cost_functions import BipartitionSimilarity

# --- setup
with open("experiments/09-noise-investigation.yaml", "r") as f:
    conf = Configuration.from_yaml(yaml.safe_load(f))
save_figs = True
data = _generate_data(conf)

for a in range(3, 13):
    conf.agreement = a
    base_folder = Path(f"results/09-noise-investigation/agreement-{a}")
    os.makedirs(base_folder, exist_ok=True)
    verbose = True

    np.random.seed(conf.seed)
    random.seed(conf.seed)

    imputation = ImputationMethod(conf.imputation_method)
    questionnaire = Questionnaire.from_euclidean(
        data.xs, noise=conf.noise, density=conf.density, seed=conf.seed,
        verbose=verbose)
    imputed_questionnaire_values = imputation(questionnaire.values)

    questionnaire_exact = Questionnaire.from_euclidean(
        data.xs, noise=0.0, density=conf.density, seed=conf.seed,
        verbose=verbose).values

    # Modified to also return the sorting index. This is needed so we can
    # sort the exact cuts as well and match the exact cuts
    # together with the noisy ones.

    def compute_cost_and_order_cuts_return_idx(bipartitions, cost_function, verbose=True):
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
    bipartitions = Cuts((imputed_questionnaire_values == 1).T)
    cost_fn = BipartitionSimilarity(bipartitions.values.T)
    cuts, idx = compute_cost_and_order_cuts_return_idx(bipartitions, cost_fn)

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

    baseline = Baseline("soe-gmm")
    y_base = baseline.predict(data.xs, questionnaire, conf.n_components)
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
