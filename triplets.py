"""
A file that is made for handling triplet data.
"""
import numpy as np
import random
from typing import Optional
from imputation import MISSING_VALUE


def distance_function(x, y):
    return np.linalg.norm(x - y)


def is_triplet(a, b, c, distances, noise=0.0, soft_threshhold: float = None, flip_noise=False, randomize_tie: bool = False, similarity: bool = False):
    """"
    Returns 1 if a is closer to b than c, 0 otherwise.
    If noise > 0, then the questions answer is set to -1 with probability noise.

    distances: ndarray of shape (num_datapoints, num_datapoints),
        precomputed distances, where distances_ij = distance between point i and j
    soft: If soft is given, the answer is set randomly if a is further away from b
        and c than soft
    flip_noise: if set to True, the answer is flipped with probability noise instead of set to 0
                (this is equivalent to calling this function with flip_noise = False and noise' = 2*noise, 
                but makes it easier to reproduce the results in the Tangles paper)
    """
    if soft_threshhold is not None and distances[a, b] > soft_threshhold and distances[a, c] > soft_threshhold:
        return MISSING_VALUE
    dist_ab = distances[a, b]
    dist_ac = distances[a, c]
    if randomize_tie and dist_ab == dist_ac:
        return int(np.random.random() < 0.5)
    else:
        if similarity:
            result = dist_ab >= dist_ac
        else:
            result = dist_ab <= dist_ac
    if noise > 0 and random.random() < noise:
        if flip_noise:
            return int(not result)
        else:
            return MISSING_VALUE
    else:
        return int(result)


def subsample_triplets_euclidean(data: np.ndarray, number_of_triplets: int, return_responses: bool = True) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns a triplet-response array with a certain number of triplets in it.
    """
    triplets = np.zeros((number_of_triplets, 3), dtype=int)
    responses = np.zeros(number_of_triplets, dtype=bool)
    for i in range(number_of_triplets):
        while True:
            # drawing indices
            i_a, i_b, i_c = random.randint(0, data.shape[0] - 1), random.randint(
                0, data.shape[0] - 1), random.randint(0, data.shape[0] - 1)
            if i_b != i_c:
                break
        a = data[i_a, :]
        b = data[i_b, :]
        c = data[i_c, :]
        triplets[i, 0] = i_a
        triplets[i, 1] = i_b
        triplets[i, 2] = i_c
        responses[i] = True if np.linalg.norm(
            a - b) < np.linalg.norm(a - c) else False
    if return_responses:
        return triplets, responses
    else:
        return unify_triplet_order(triplets, responses), None


def unify_triplet_order(triplets: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """
    Takes in an array of triplets and responses, and reorders the triplets such that it always
    holds that a triplet has the meaning

    triplet[0] is closer to triplet[1] than to triplet[2]
    """
    wrong_order = np.logical_not(responses)
    # swap those in wrong order
    triplets[wrong_order, 1], triplets[wrong_order,
                                       2] = triplets[wrong_order, 2], triplets[wrong_order, 1]
    return triplets


def triplets_to_majority_neighbour_cuts(triplets: np.ndarray, radius: float = 1, randomize_tie: bool = False, sigmoid_scale: float = None, seed=None) -> np.ndarray:
    """
    Calculates the majority neighbour cuts for a set of triplets.
    If a sigmoid scale is given, the cuts aren't made binary (if b has appeared more often in middle 
    position than in right position it is included), but the difference between b being in middle vs b 
    being right is used as input to a sigmoid to then get a probability of b being in the cut or not
    (so b being in the middle often => b will have a high chance of getting included in the cut). 

    Returns array of triplets.
    Triplets are in array format, such that the following is true:

    Triplets[0] is closer to Triplets[1] than Triplets[2].
    """
    if seed is not None:
        np.random.seed(seed)
    max_point = triplets.max()
    first_positions_points = np.unique(triplets[:, 0])

    cuts = np.zeros((max_point + 1, first_positions_points.size))
    for a in first_positions_points:
        triplets_starting_with_a = triplets[triplets[:, 0] == a, :]
        counts_b_is_closer = np.bincount(
            triplets_starting_with_a[:, 1], minlength=max_point + 1)
        counts_b_is_farther = np.bincount(
            triplets_starting_with_a[:, 2], minlength=max_point + 1)
        if randomize_tie:
            ties = counts_b_is_closer == counts_b_is_farther
            counts_b_is_closer[ties] += np.random.choice(
                2, counts_b_is_closer.shape)[ties]

        # Points are in the same partition if they are more often closer to point than farther
        if sigmoid_scale is None:
            cut = radius * counts_b_is_closer > counts_b_is_farther
        else:
            cut_probabilities = _sigmoid(
                radius * counts_b_is_closer - counts_b_is_farther, sigmoid_scale)
            draws = np.random.uniform(size=cut_probabilities.shape)
            cut = cut_probabilities >= draws
        cut[a] = True
        cuts[:, a] = cut
    return cuts


def _sigmoid(x, scale):
    return 1 / (1 + np.exp(-x * scale))
