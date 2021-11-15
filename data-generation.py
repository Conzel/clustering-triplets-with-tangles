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
from questionnaire import generate_questionnaire, Questionnaire

plt.style.use('ggplot')

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

# Creating the questionnaire from the data
questionnaire = generate_questionnaire(data).values

# Interpreting the questionnaires as cuts and computing their costs
bipartitions = data_types.Cuts((questionnaire == 1).T)
cuts = utils.compute_cost_and_order_cuts(bipartitions, partial(
    cost_functions.mean_manhattan_distance, questionnaire, None))

# Building the tree, contracting and calculating predictions
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

# Plotting the hard clustering
plotting.plot_hard_predictions(data=data, ys_predicted=ys_predicted,
                               path=None)
