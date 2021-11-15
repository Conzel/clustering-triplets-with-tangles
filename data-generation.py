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
import sklearn
import yaml
from src.loading import load_GMM
from functools import partial
from questionnaire import generate_questionnaire, Questionnaire

plt.style.use('ggplot')

# Parameters for data generation
config_file = "experiments/01-tangle-algorithm-crashes.yaml"
with open(config_file, 'r') as f:
    data = yaml.safe_load(f)

# ---- loading parameters ----
seed = data["seed"]
np.random.seed(seed)
means = data["means"]
stds = data["stds"]
num_clusters = len(means)
n = data["n"]
total_num_samples = n * num_clusters
agreement = data["agreement"]
figure_output_path = Path(data["figure_output_path"])
num_distance_function_samples = data["num_distance_function_samples"]

# Data generation
xs, ys = load_GMM(blob_sizes=[n] * num_clusters,
                  blob_centers=means, blob_variances=stds, seed=seed)
data = data_types.Data(xs=xs, ys=ys)

# Creating the questionnaire from the data
questionnaire = generate_questionnaire(data).values

# Interpreting the questionnaires as cuts and computing their costs
bipartitions = data_types.Cuts((questionnaire == 1).T)
cuts = utils.compute_cost_and_order_cuts(bipartitions, partial(
    cost_functions.mean_manhattan_distance, questionnaire, num_distance_function_samples))

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

# evaluate hard predictions
if data.ys is not None:
    ARS = sklearn.metrics.adjusted_rand_score(data.ys, ys_predicted)
    NMI = sklearn.metrics.normalized_mutual_info_score(data.ys, ys_predicted)

    print('Adjusted Rand Score: {}'.format(np.round(ARS, 4)), flush=True)
    print('Normalized Mutual Information: {}'.format(
        np.round(NMI, 4)), flush=True)

# Plotting the hard clustering
plotting.plot_hard_predictions(data=data, ys_predicted=ys_predicted,
                               path=None)
