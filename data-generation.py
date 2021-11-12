# Allows us to import tangles modules
import sys
import os
from pathlib import Path
sys.path.append("./tangles")

# other imports
import math
import matplotlib.pyplot as plt
import numpy as np
import src.data_types as data_types
import src.utils as utils
import src.tree_tangles as tree_tangles
import src.cost_functions as cost_functions
import src.plotting as plotting
from src.loading import load_GMM
from functools import partial

# Parameters for data generation
seed = 314159628
np.random.seed(seed)
means = [[0, -10], [-9, 7], [9, 5], [-7, -9], [-10, 0]]
stds = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
num_clusters = len(means)
n = 10
total_num_samples = n * num_clusters
agreement = 5
figure_output_path = Path("./figures/full-triplets")

# Data generation
xs, ys = load_GMM(blob_sizes=[n] * num_clusters,
                  blob_centers=means, blob_variances=stds, seed=seed)
data = data_types.Data(xs=xs, ys=ys)


def distance_function(x, y): return np.linalg.norm(x - y)


# some plotting to show that sampling works
for k in range(num_clusters):
    points_in_k = data.xs[data.ys == k]
    #
    plt.scatter(points_in_k[:, 0], points_in_k[:, 1], label=str(k))
plt.savefig(figure_output_path.joinpath("samples.png"))


def is_triplet(a, b, c, dist=distance_function):
    """"
    Returns 1 if a is closer to b than c, 0 otherwise
    """
    return int(dist(a, b) <= dist(a, c))

# This function takes in a parameter k and a set of values to choose from.
# It returns every possible subset of length k from the set of values


def generate_k_subsets(values, k):
    subsets = []
    for i in range(len(values)):
        if k == 1:
            subsets.append([values[i]])
        else:
            for subset in generate_k_subsets(values[i + 1:], k - 1):
                subsets.append([values[i]] + subset)
    return subsets


# Generate the triplet table
question_set = generate_k_subsets(list(range(total_num_samples)), 2)
assert len(question_set) == math.comb(total_num_samples, 2)

# Iterate over all points and answer all questions for them.
# The questionnaire contains all answers for all questions.
# These represent bipartitions on our data. We can now assign a Hamming-Cost
# to each bipartition.
questionnaire = np.zeros((total_num_samples, len(question_set)))
for i in range(total_num_samples):
    a = data.xs[i]
    answers = []
    for question in question_set:
        b = data.xs[question[0]]
        c = data.xs[question[1]]
        answer = is_triplet(a, b, c)
        answers.append(answer)
    questionnaire[i] = np.array(answers)

bipartitions = data_types.Cuts((questionnaire == 1).T)
cuts = utils.compute_cost_and_order_cuts(bipartitions, partial(
    cost_functions.mean_manhattan_distance, questionnaire, None))

print("Building the tangle search tree", flush=True)
tangles_tree = tree_tangles.tangle_computation(cuts=cuts,
                                               agreement=agreement,
                                               verbose=0  # print everything
                                               )

contracted = tree_tangles.ContractedTangleTree(tangles_tree)
contracted.prune(5)

contracted.calculate_setP()

# soft predictions
weight = np.exp(-utils.normalize(cuts.costs))
tree_tangles.compute_soft_predictions_children(
    node=contracted.root, cuts=bipartitions, weight=weight, verbose=3)

ys_predicted, _ = utils.compute_hard_predictions(contracted, cuts=bipartitions)

# show hard clustering
plotting.plot_hard_predictions(data=data, ys_predicted=ys_predicted,
                               path=figure_output_path)

# plot soft predictions
plotting.plot_soft_predictions(data=data,
                               contracted_tree=contracted,
                               eq_cuts=bipartitions.equations,
                               path=figure_output_path)
