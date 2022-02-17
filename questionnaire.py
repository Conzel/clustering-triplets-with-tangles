"""
File containing functions and class definitions for working with triplets in questionnaire form. 
"""
from __future__ import annotations
import math
import random
import re
import networkx as nx
from operator import itemgetter

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import DistanceMetric
from tqdm import tqdm

from data_generation import clean_bipartitions

MISSING_VALUE = -1


def distance_function(x, y):
    return np.linalg.norm(x - y)


def is_triplet(a, b, c, distances, noise=0.0, soft_threshhold: float = None, flip_noise=False):
    """"
    Returns 1 if a is closer to b than c, 0 otherwise.
    If noise > 0, then the questions answer is set to -1 with probability noise.

    distances: ndarray of shape (num_datapoints, num_datapoints),
        precomputed distances, where distances_ij = distance between point i and j
    soft: If soft is given, the answer is set randomly if a is further away from b
        and c than soft
    flip_noise: if set to True, the answer is flipped with probability noise instead of set to 0
                (this is equivalent to calling this functionwith flip_noise = False and noise' = 2*noise, 
                but makes it easier to reproduce the results in the Tangles paper)
    """
    if soft_threshhold is not None and distances[a, b] > soft_threshhold and distances[a, c] > soft_threshhold:
        return MISSING_VALUE
    if noise > 0 and random.random() < noise:
        if flip_noise:
            return int(distances[a, b] > distances[a, c])
        else:
            return MISSING_VALUE
    else:
        return int(distances[a, b] <= distances[a, c])


def generate_k_subsets(values: list, k: int) -> "list[list]":
    """
    This function takes in a parameter k and a set of values to choose from.
    It returns every possible subset of length k from the set of values
    """
    subsets = []
    for i in range(len(values)):
        if k == 1:
            subsets.append([values[i]])
        else:
            for subset in generate_k_subsets(values[i + 1:], k - 1):
                subsets.append([values[i]] + subset)
    return subsets


def generate_question_set(num_datapoints: int, density=1.0):
    max_amount_of_questions = math.comb(num_datapoints, 2)
    if density > 0.1:  # very rough, we could use a better metric here
        question_set = generate_k_subsets(list(range(num_datapoints)), 2)
        assert len(question_set) == max_amount_of_questions
        # Removing questions from the set of all questions
        if density < 1.0:
            actual_amount_of_questions = math.floor(
                max_amount_of_questions * density)
            idx = random.sample(range(max_amount_of_questions),
                                actual_amount_of_questions)
            question_set = itemgetter(*idx)(question_set)

        return question_set

    # very sparse set, we are better off directly sampling numbers
    else:
        total_questions = math.comb(num_datapoints, 2) * density
        sampled_questions = set()
        num_sampled = 0
        while num_sampled < total_questions:
            b, c = random.randint(
                0, num_datapoints - 1), random.randint(0, num_datapoints - 1)
            if b > c and (b, c) not in sampled_questions:
                sampled_questions.add((b, c))
                num_sampled += 1
            else:
                continue
        return sampled_questions


class ImputationMethod():
    """
    Method for imputing missing data on binary data arrays (questionnaires in this case).
    Imputation methods do not change the data, they copy it (although this could be changed in
    the future if performance demands it.)

    RANDOM: Fills in a random value.
    k-NN: Fills in the value with the mean of the most common k neighbours, where k is an int.
    MEAN: Fills in the value with the dataset mean.
    """

    def __init__(self, method_name: str):
        """
        Initiates the imputation method. Additional arguments are
        given through the constructor and might be required for methods.
        Neighbours imputation f.e. needs to know how many neighbours to use.
        """
        self.method_name = method_name
        self.method = ImputationMethod._parse_imputation(method_name)

    def __str__(self) -> str:
        return "ImputationMethod(" + self.method_name + ")"

    def __repr__(self) -> str:
        return self.__str__()

    def _impute_random(data: np.ndarray):
        """
        Imputes missing values with a random value.
        """
        imputed_data = data.copy()
        imputed_data[imputed_data == -
                     1] = np.random.randint(0, 2, imputed_data[imputed_data == MISSING_VALUE].shape)
        return imputed_data

    def _impute_knn(data: np.ndarray, k: int):
        """
        Imputes missing values with the mean value of the k nearest neighbours. 
        Coinflip decides on 0.5.
        """
        print("Imputing via knn")
        imputer = KNNImputer(n_neighbors=k, missing_values=MISSING_VALUE)
        imputed_data = imputer.fit_transform(data)
        # removing the 0.5 values with random values
        imputed_data[imputed_data == 0.5] = np.random.randint(
            0, 2, imputed_data[imputed_data == 0.5].shape)
        return np.around(imputed_data)

    def _impute_mean(data: np.ndarray):
        """
        Imputes missing values with the mean value of the column.

        According to:
        https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
        """
        imputed_data = data.copy()
        imputed_data[imputed_data == MISSING_VALUE] = np.nan
        col_mean = np.nanmean(imputed_data, axis=0)
        inds = np.where(np.isnan(imputed_data))
        imputed_data[inds] = np.take(col_mean, inds[1])
        return imputed_data

    def _parse_imputation(imputation_method: str):
        """
        Parses the imputation method from a string.
        """
        knn_regex = re.search("(\d+)-NN", imputation_method)
        if imputation_method.lower() == "random":
            return ImputationMethod._impute_random
        elif knn_regex is not None:
            k = knn_regex.group(1)
            return lambda x: ImputationMethod._impute_knn(x, int(k))
        elif imputation_method.lower() == "mean":
            return ImputationMethod._impute_mean

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Imputes given data with the method used on construction.

        INPUT:
            data: np.ndarray
            Binary data (consisting of 0-1 values) in a nxm array to impute. Missing values are marked
            with a -1.
        OUTPUT:
            Imputed data of form nxw with w < n. No more values are allowed to be -1,
            these have been replaced with an imputation.
        """
        return self.method(data)


class Questionnaire():
    """
    Data type representing a questionnaire. 

    The questionnaire field is a ndarray. It contains the points a as rows. Each column stands for a pair {b,c}. The value of the cell
    is 1 if a is closer to b than c, 0 otherwise.

    The labels field associates the columns of the questionnaire with pairs {b,c}. 
    The entry at position i contains the tuple (j,k) associated with the indices of the
    pair {b,c} in the origin column i.
    """

    def __init__(self, questionnaire: np.ndarray, labels: "list[tuple]") -> None:
        assert len(labels) == questionnaire.shape[1]
        self.values = questionnaire
        self.labels = labels

    def __str__(self) -> str:
        return "Questionnaire(\n" + str(self.values) + "\n)"

    def __repr__(self) -> str:
        return self.__str__()

    def subsample_triplets_euclidean(data: np.ndarray, number_of_triplets: int, return_responses: bool = True) -> tuple(np.ndarray, np.ndarray):
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

    def from_euclidean(data: np.ndarray, noise=0.0, density=1.0, verbose=True, seed=None, soft_threshhold: float = None, imputation_method: str = None, flip_noise: bool = False) -> Questionnaire:
        """
        Generates a questionnaire from euclidean data. 
        data is a nxm ndarray with n points and m features. 

        For information on the other arguments, see "_generate_questionnaire".
        """
        metric = DistanceMetric.get_metric("euclidean")
        # cached distances for all points
        distances = metric.pairwise(data)
        return _generate_questionnaire(distances, noise, density, verbose, seed, soft_threshhold, imputation_method, flip_noise)

    @staticmethod
    def from_graph(data: nx.Graph, noise=0.0, density=1.0, verbose=True, seed=None, soft_threshhold: float = None, imputation_method: str = None, flip_noise: bool = False):
        """
        Generates a questionnaire from a graph. 
        data is any nx graph.

        For information on the other arguments, see "_generate_questionnaire".
        """
        distances = nx.floyd_warshall_numpy(data)
        return _generate_questionnaire(distances, noise, density, verbose, seed, soft_threshhold, imputation_method, flip_noise)

    def throwout(self, threshhold: float):
        """
        Throws out column that has equal or more than threshhold% corrupted values.

        Returns a new questionnaire and doesn't change the old one.
        """
        new_vals, new_labels = Questionnaire._throwout_vals_labels(
            threshhold, self.values, self.labels)
        return Questionnaire(new_vals, new_labels)

    def _throwout_vals_labels(threshhold: float, values: np.ndarray, labels: "list[tuple]") -> "tuple[np.ndarray, list[tuple]]":
        """
        Computes new values and labels according to the procedure described in throwout.
        """
        corrupted_cols = (values == MISSING_VALUE).mean(axis=0) >= threshhold
        labels = np.array(labels)[~corrupted_cols]
        labels = [tuple(x) for x in labels]
        return values[:, ~corrupted_cols], labels

    def from_bool_array(triplets, responses) -> Questionnaire:
        """
        Generates a questionnaire from triplets and responses 
        given as bool array. This function is the opposite of 
        'triplets_to_bool_array'
        """
        # Eventually we could have more points but those don't have
        # any triplet information
        n_points = triplets.max() + 1
        values = np.zeros((n_points, math.comb(n_points, 2)))
        values[values == 0] = MISSING_VALUE
        labels = [0] * (math.comb(n_points, 2))
        for i in range(triplets.shape[0]):
            a, b, c = triplets[i]
            if b > c:
                b, c = c, b
            if b == c:
                raise ValueError(
                    "Triplet information on distance of point to itself.")
            # invariant: c > b
            val = 1 if responses[i] else 0
            values[a][Questionnaire.triplet_to_pos(b, c, n_points)] = val

        for b in range(n_points):
            for c in range(b, n_points):
                labels[Questionnaire.triplet_to_pos(b, c, n_points)] = (b, c)

        return Questionnaire(values, labels)

    def triplet_to_pos(b, c, n_points):
        """
        Returns column position of the triplet information (correspondence
        of triplet and question).
        """
        return int(b * (n_points - 1) + (c - 1) - (b**2 + b) / 2)

    def from_bipartitions(xs: np.ndarray) -> Questionnaire:
        """
        Takes in an ndarray that comes from a any dataset that produces bipartitions
        (such as the mindset case, refer to Klepper et al. p.14). 
        Cleans the data (removing degenerate columns) and returns a questionnaire.

        To inquiry about how we can interpret a mindset setup as a triplet
        setting, see experiments/15_mindset_datasets.ipynb.
        """
        xs = clean_bipartitions(xs)
        columns = xs.shape[1]
        labels = []
        for i in range(columns):
            col = xs[:, i]
            if col.dtype == "bool":
                possible_bs = np.where(col)[0]
                possible_cs = np.where(col)[0]
            elif col.dtype == "int":
                possible_bs = np.where(col == 1)[0]
                possible_cs = np.where(col == 0)[0]
            else:
                raise ValueError("Mindset data must be boolean or integer.")
            while True:
                b, c = np.random.choice(
                    possible_bs), np.random.choice(possible_cs)
                if (b, c) not in labels:
                    labels.append((b, c))
                    break

        return Questionnaire(xs, labels)

    def to_bool_array(self, return_responses: bool = True) -> tuple(np.ndarray, np.ndarray):
        """
        Transforms triplet value matrix to a list representing the triplets,
        conforming to the interface in David Künstles cblearn package under
        'triplet-array'. Returns either only triplets (return_responses = False),
        or responses as well (return_responses = True).

        Entries in the questionnaire that are missing (have value MISSING_VALUE)
        are left out of the array.

        https://cblearn.readthedocs.io/en/latest/generated_examples/triplet_formats.html?highlight=matrix

        To Quote:

        > In the array format, the constraints are encoded by the index order.
        >
        > [responses = False]
        > triplet = triplets_ordered[0]
        > print(f"The triplet {triplet} means, that object {triplet[0]} (1st) should be "
        >     f"embedded closer to object {triplet[1]} (2nd) than to object {triplet[2]} (3th).")
        >
        > [responses = True]
        > triplets_boolean, answers_boolean = check_query_response(triplets_ordered, result_format='list-boolean')
        > print(f"Is object {triplets_boolean[0, 0]} closer to object {triplets_boolean[0, 1]} "
        >     f"than to object {triplets_boolean[0, 2]}? {answers_boolean[0]}.")
        >
        Returns a tuple where the first part is the triplet array, the second part is the answer 
        array (all as specified above). Answer array might be None if return_responses = False.
        """
        triplets = []
        responses = self.values.flatten()
        labels_np = np.array(self.labels)
        num_questions = labels_np.shape[0]
        for a in range(self.values.shape[0]):
            a_array = np.repeat(a, num_questions).reshape(-1, 1)
            triplets.append(np.hstack((a_array, labels_np)))
        triplet_array = np.concatenate(triplets)
        valid_mask = ~(responses == MISSING_VALUE)

        triplet_array = triplet_array[valid_mask, :]
        responses = responses[valid_mask] == 1
        if return_responses:
            return triplet_array, responses
        else:
            return unify_triplet_order(triplet_array, responses), None

    def subset(self, n, seed=None) -> Questionnaire:
        """
        Returns a questionnaire that only has n randomly drawn triplets left,
        the rest is set to missing_value.

        We draw triplets with replacement, so we might end up with 
        a few less samples than there are actually present.
        """
        np.random.seed(seed)
        idx_rows = np.random.choice(self.values.shape[0], n, replace=True)
        idx_cols = np.random.choice(self.values.shape[1], n, replace=True)
        mask = np.ones_like(self.values)
        mask[idx_rows, idx_cols] = 0
        values = self.values.copy()
        values[mask == 1] = MISSING_VALUE
        return Questionnaire(values, self.labels)

    def impute(self, imputation_method: str) -> Questionnaire:
        """
        Imputes the questionnaire with the given method.
        Afterwards, the questionnaire is guaranteed to not have any 
        rows left with missing values. 
        This might change the shape of the questionnaire, as completely
        corrupted columns will be thrown out.

        INPUT:
            imputation_method: str
                The method to use for imputation. See ImputationMethod for possible
                values to use.
        OUTPUT:
            Questionnaire with imputed values.
        """
        if imputation_method is None:
            raise ValueError("'None' is not a valid imputation method.")
        imputation_method = ImputationMethod(imputation_method)
        cleaned_values, cleaned_labels = Questionnaire._throwout_vals_labels(
            1, self.values, self.labels)
        return Questionnaire(imputation_method(cleaned_values), cleaned_labels)


def create_log_function(verbose):
    if verbose:
        return lambda x: print(x)
    else:
        return lambda _: None


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


def majority_neighbour_cuts(triplets: np.ndarray, radius: float = 1, randomize_tie: bool = False, sigmoid_scale: float = None, seed=None):
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


def _generate_questionnaire(distances: np.ndarray, noise: float = 0.0, density: float = 1.0, verbose: bool = True,
                            seed: int = None, soft_threshhold: float = None, imputation_method: str = None, flip_noise: bool = False) -> Questionnaire:
    """
    Generates a questionnaire for the given data. This is agnostic of the data source, 
    as any distance matrix can be passed (which can arise from euclidean data, a graph or anything else).

    Input: 
    - distances: np.ndarray
      Distance matrix of size data_points x data_points. distances[i, j] is the distance
      from point i to point j.
    - noise: float
      The percentage of noise to add to the questionnaire. 0 means all answers are truthful, 
      1 means all answers are random.
    - density: float
      The peprcentage of questions we know the answers. 1.0 means we know the answers to all questions, 
      0.0 means we don't know any of them. This amounts to the number of columns that will be contained 
      in the questionnaire.
    - seed: int
    For soft and flip_noise, see "is_triplet".

    --------------------------------------------------------------------------------------------
    Output: Questionnaire
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    assert 0 <= noise <= 1
    assert 0 <= density <= 1

    log = create_log_function(verbose)
    num_datapoints = distances.shape[0]
    log("Generating questionnaire...")

    log("Generating question set...")
    question_set = generate_question_set(num_datapoints, density)

    # Iterate over all points and answer all questions for them.
    # The questionnaire contains all answers for all questions.
    log("Filling out questionnaire...")
    questionnaire = np.zeros((num_datapoints, len(question_set)))

    for i in tqdm(range(num_datapoints), disable=not verbose):
        a = i
        answers = np.zeros(len(question_set))
        for j, question in enumerate(question_set):
            b = question[0]
            c = question[1]
            answer = is_triplet(a, b, c, distances, noise=noise,
                                soft_threshhold=soft_threshhold, flip_noise=flip_noise)
            answers[j] = answer
        questionnaire[i] = np.array(answers)

    questionnaire = Questionnaire(
        questionnaire, list(map(tuple, question_set)))
    if imputation_method is not None:
        questionnaire = questionnaire.impute(
            imputation_method=imputation_method)

    return questionnaire
