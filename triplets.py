"""
A file that is made for handling triplet data.
"""
import numpy as np
import random
from typing import Optional
from imputation import MISSING_VALUE
from sklearn.neighbors import DistanceMetric


def distance_function(x, y):
    return np.linalg.norm(x - y)


def is_triplet(a, b, c, distances, noise=0.0, soft_threshhold: Optional[float] = None, flip_noise=False, randomize_tie: bool = False, similarity: bool = False):
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


def _lens_distance(triplets: np.ndarray, most_central_triplet: np.ndarray, x: int, y: int, normalize: bool = False) -> int:
    """
    See lens_distance_matrix.
    Responses has been replaced by just an array of the most central triplet for each triplet
    for computational reasons.

    If normalize is set to true, divides the distances by the number of statements containing both x and y
    """
    contains_x_y = np.logical_and(
        np.any(triplets == x, axis=1), np.any(triplets == y, axis=1))
    other_is_most_central = np.logical_and(
        most_central_triplet[contains_x_y] != x, most_central_triplet[contains_x_y] != y)
    # we can take the sum directly, as we assume uniform randomly sampled triplets
    dist = other_is_most_central.sum()
    if contains_x_y.sum() == 0:
        # maximum distance, we don't use the
        return triplets.max() + 1
    elif normalize:
        return dist / contains_x_y.sum()
    else:
        return dist


def remove_outliers(triplets: np.ndarray, responses: np.ndarray, outliers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes the indices in the array outliers from the triplets and responses arrays.
    This induces renaming of the elements in the triplet array, such that the elements
    are still contigous after the renaming.

    Outliers shall be a 1-dimensional array that contains the names of all the outliers.

    f.e. if we have 5 elements, and element 2 should be removed, the namings will be:
    1 -> 1, 2 -> removed, 3 -> 2, 4 -> 3, 5 -> 4

    Returns cleaned triplets and responses arrays.
    """
    assert len(outliers.shape) == 1

    outliers_list = list(outliers)
    outliers_list.append(triplets.max() + 1)

    # getting replacements for numbers
    replacements = {}
    for (i, o) in enumerate(outliers_list):
        if i == 0:
            start = -1
        else:
            start = outliers[i - 1]
        for j in range(start + 1, o):
            replacements[j] = j - i
        replacements[o] = None

    contains_outlier = np.any(
        np.any((triplets[None, :, :] == outliers[:, None, None]), axis=0), axis=1)
    triplets_without_outliers = triplets[~contains_outlier, :]
    responses_without_outliers = responses[~contains_outlier]

    replacev = np.vectorize(lambda x: replacements[x])
    triplets_without_outliers_renamed = replacev(triplets_without_outliers)
    return triplets_without_outliers_renamed, responses_without_outliers


class LensMetric():
    """
    SkLearn compatible metric using the lens distance between points.
    Always uses euclidean distances
    """

    def pairwise(self, xs: np.ndarray):
        """
        Returns the pairwise lens distance between all points. Uses a euclidean
        metric to determine distances.
        """
        n_points = xs.shape[0]
        point_to_point_dist = DistanceMetric.get_metric(
            'euclidean').pairwise(xs)
        lens_dists = np.zeros((n_points, n_points))
        for i in range(0, n_points):
            for j in range(i + 1, n_points):
                radius = point_to_point_dist[i, j]
                in_lens = np.logical_and(
                    point_to_point_dist[:, i] < radius, point_to_point_dist[:, j] < radius)
                lens_dists[i, j] = np.sum(in_lens)
        return lens_dists + lens_dists.T

    def pairwise_triplets(self, triplets: np.ndarray, responses: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Based on the dissertation of Kleindessner, 2017, available here:
        https://hsbiblio.uni-tuebingen.de/xmlui/bitstream/handle/10900/77452/thesis_Kleindessner.pdf

        We interpret and calculate a distance as follows:
        If the lens (all points c, such that c is the most central object of (x,c,y))
        of x and y contain a lot of points, they are further away than if the lens contains only a few points.

        If normalize is set to true, divides each distance by the total amount of objects 
        that compromised both x and y to begin with. The dissertation of Kleindessner, p.84.

        Calculates the pairwise lens distance between all points.
        """
        assert triplets.shape[1] == 3
        assert triplets.shape[0] == responses.shape[0]
        n_points = triplets.max() + 1
        distances = np.zeros((n_points, n_points))
        most_central_triplet = np.take_along_axis(
            triplets, responses[:, None], axis=1)
        for i in range(0, n_points):
            for j in range(i + 1, n_points):
                distances[i, j] = _lens_distance(
                    triplets, most_central_triplet, i, j, normalize=normalize)
        return distances + distances.T

    def outside_point(self, xs: np.ndarray, x: int, z: np.ndarray):
        """
        Calculates the lens distance between a point x and a point z, while
        z is a point that is not contained in the dataset.
        """
        if len(z.shape) == 1:
            z = z[None, :]
        n_points = xs.shape[0]
        euclidean = DistanceMetric.get_metric('euclidean')
        point_to_point_dist = euclidean.pairwise(xs)
        z_and_points_dist = euclidean.pairwise(xs, z)
        radius = z_and_points_dist[x, 0]
        in_lens = np.logical_and(
            point_to_point_dist[:, x] < radius, z_and_points_dist[:, 0] < radius)
        return in_lens.sum()


def subsample_triplets(data: np.ndarray, number_of_triplets: int, metric=DistanceMetric.get_metric('euclidean'), return_mostcentral: bool = False, seed: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns a triplet-response array with a certain number of triplets in it.

    If return_mostcentral is set to True, arrays returned are sucht that the following holds:

    Out of the objects triplets[i][0], triplets[i][1], triplets[i][2], triplets[i][responses[i]]
    is the most central object ouf of them.

    Else they are such that this holds:
    Is triplet[i][0] closer to triplet[i][1] than to triplet[i][2]? responses[i]
    """
    if seed is not None:
        random.seed(seed)
    triplets = np.zeros((number_of_triplets, 3), dtype=int)
    if return_mostcentral is True:
        responses = np.zeros(number_of_triplets, dtype=int)
    else:
        responses = np.zeros(number_of_triplets, dtype=bool)
    dists = metric.pairwise(data)
    for i in range(number_of_triplets):
        while True:
            # drawing indices
            a, b, c = random.randint(0, data.shape[0] - 1), random.randint(
                0, data.shape[0] - 1), random.randint(0, data.shape[0] - 1)
            if a != b and a != c and b != c:
                break
        triplets[i, 0] = a
        triplets[i, 1] = b
        triplets[i, 2] = c
        if return_mostcentral is True:
            responses[i] = _most_central(a, b, c, dists)
        else:
            responses[i] = True if dists[a, b] < dists[a, c] else False
    return triplets, responses


def _most_central(a: int, b: int, c: int, distances: np.ndarray) -> int:
    """
    Returns 0 if a is the most central object out of (a,b,c), 1 if b is the most central object,
    else 2.

    Assumes that distance matrix is symmetric.
    """
    assert np.all(distances == distances.T)
    dists = [distances[a, b] + distances[a, c], distances[b, c] +
             distances[b, a], distances[c, a] + distances[c, b]]
    return np.argmin(dists).item()


def unify_triplet_order(triplets: np.ndarray, responses: np.ndarray) -> np.ndarray:
    """
    Takes in an array of triplets and responses, and reorders the triplets such that it always
    holds that a triplet has the meaning

    triplet[0] is closer to triplet[1] than to triplet[2]
    """
    triplets = triplets.copy()
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
    # TODO: This is mostly duplicated code as the stochastic matrix method below. Maybe
    # we can bring those two together and remove some code.
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


def majority_neighbours_count_matrix(triplets: np.ndarray) -> np.ndarray:
    """
    Returns a matrix where the entry i,j contains how often the point j was closer to i than it was farther.
    Assume we have the triplets (a,b,c), (a,b,e), (a,b,f), (a,l,b)
    then the matrix would have the value 4-1 = 3 for the entry (a,b)
    """
    max_point = triplets.max()
    first_positions_points = np.unique(triplets[:, 0])

    cuts = np.zeros((max_point + 1, first_positions_points.size))
    for a in first_positions_points:
        triplets_starting_with_a = triplets[triplets[:, 0] == a, :]
        counts_b_is_closer = np.bincount(
            triplets_starting_with_a[:, 1], minlength=max_point + 1)
        counts_b_is_farther = np.bincount(
            triplets_starting_with_a[:, 2], minlength=max_point + 1)
        cuts[:, a] = counts_b_is_closer - counts_b_is_farther
    np.fill_diagonal(cuts, max_point)
    return cuts


def triplets_to_stochastic_matrix_cuts(triplets: np.ndarray, threshhold: float = 0.25, iterations: int = 2, seed=None):
    A = majority_neighbours_count_matrix(triplets)
    n = triplets.shape[0]
    A = ((A + n) / (2 * n))
    for _ in range(iterations - 1):
        A = A @ A
    return A > threshhold


def _sigmoid(x, scale):
    return 1 / (1 + np.exp(-x * scale))
