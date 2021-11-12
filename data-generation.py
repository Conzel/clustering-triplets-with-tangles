# Allows us to import tangles modules
import sys
import os
sys.path.append("./tangles")

# other imports
import math
import matplotlib.pyplot as plt
import numpy as np
import src.data_types as data_types
import src.utils as utils
import src.tree_tangles as tree_tangles
import src.cost_functions as cost_functions
from functools import partial

# Parameters for data generation
np.random.seed(314159628)
means = [-1, 0, 1]
stds = [0.1, 0.1, 0.1]
num_clusters = len(means)
n = 10
dim_points = 2
total_num_samples = n * num_clusters
agreement = 3


def distance_function(x, y): return np.linalg.norm(x - y)


# Generate a set N of n datapoints from a mixture of gaussians. Each mixture component represents
# one ground truth cluster.
gaussians = np.random.normal(means, stds, (n, dim_points, num_clusters))
# unseparated clusters
gaussians_mixed = gaussians.reshape((total_num_samples, dim_points))

# some plotting to show that sampling works
for k in range(num_clusters):
    plt.scatter(gaussians[:, 0, k], gaussians[:, 1, k])
plt.show()


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
    a = gaussians_mixed[i]
    answers = []
    for question in question_set:
        b = gaussians_mixed[question[0]]
        c = gaussians_mixed[question[1]]
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

colors = ['red', 'blue', 'green', 'black']
# some plotting to show that sampling works
for i in range(total_num_samples):
    plt.scatter(gaussians_mixed[i, 0],
                gaussians_mixed[i, 1], c=colors[ys_predicted[i]])
plt.show()
