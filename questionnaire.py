"""
File containing functions and class definitions for working with triplets in questionnaire form. 
"""
from __future__ import annotations
import math
from hierarchies import HierarchyTree
from imputation import MISSING_VALUE, ImputationMethod
import random
from typing import Optional
import networkx as nx
from operator import itemgetter
from triplets import lens_distance_matrix, unify_triplet_order, is_triplet

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import DistanceMetric
from tqdm import tqdm

from data_generation import clean_bipartitions


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


class Questionnaire():
    """
    Data type representing a questionnaire. 

    The questionnaire field is a ndarray. It contains the points a as rows. Each column stands for a pair {b,c}. The value of the cell
    is 1 if a is closer to b than c, 0 otherwise.

    The labels field associates the columns of the questionnaire with pairs {b,c}. 
    The entry at position i contains the tuple (j,k) associated with the indices of the
    pair {b,c} in the origin column i.
    """

    def __init__(self, questionnaire: np.ndarray, labels: list[tuple[int, int]]) -> None:
        assert len(labels) == questionnaire.shape[1]
        self.values = questionnaire
        self.labels = labels

    def __str__(self) -> str:
        return "Questionnaire(\n" + str(self.values) + "\n)"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_metric(data: np.ndarray, metric=None, noise=0.0, density=1.0, verbose=True, seed=None, soft_threshhold: float = None, imputation_method: str = None, flip_noise: bool = False) -> Questionnaire:
        """
        Generates a questionnaire from euclidean data. 
        data is a nxm ndarray with n points and m features. 

        If metric is None, we use the euclidean metric. Else, an sklearn-compatible metric
        can be passed. It only has to posess a 'pairwise' function, that takes in an array
        and returns a gram matrix.

        For information on the other arguments, see "_generate_questionnaire".
        """
        if metric is None:
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

    @staticmethod
    def from_hierarchy(hierarchy: HierarchyTree, labels: np.ndarray, randomize_ties: bool = True, noise=0.0, density=1.0, verbose=True, seed=None, soft_threshhold: float = None, imputation_method: str = None, flip_noise: bool = False) -> Questionnaire:
        similarities = np.zeros((labels.shape[0], labels.shape[0]))
        for i in range(labels.shape[0]):
            for j in range(i, labels.shape[0]):
                similarities[i, j] = hierarchy.closest_ancestor_level(
                    labels[i], labels[j])
        similarities = similarities + similarities.T
        return _generate_questionnaire(similarities, noise, density, verbose, seed, soft_threshhold, imputation_method, flip_noise, randomize_ties=randomize_ties, similarity=True)

    @staticmethod
    def from_most_central_triplets(triplets: np.ndarray, responses: np.ndarray, randomize_ties: bool = True, noise=0.0, density=1.0, verbose=True, seed=None, soft_threshhold: float = None, imputation_method: str = None, flip_noise: bool = False) -> Questionnaire:
        """
        Generates a questionnaire from similarity triplets.

        Each row of the triplet array corresponds to three datapoints that have been shown 
        to a participant. The responses indicate the most central datapoint of each triplet.
        """
        return _generate_questionnaire(lens_distance_matrix(triplets, responses), noise, density, verbose, seed, soft_threshhold, imputation_method, flip_noise, randomize_ties=randomize_ties)

    @staticmethod
    def from_bool_array(triplets, responses, self_fill: bool = True) -> Questionnaire:
        """
        Generates a questionnaire from triplets and responses given as bool array. 
        This function is the opposite of 'triplets_to_bool_array'

        self_fill:
            If set to true, we also fill in self-information with the correct values: 
            A point is always closer to itself than to another point. This is reflected in the returned questionnaire,
            even if not provided in the original triplet information.

        The arguments are as received from the corresponding cblearn functions 
        when called with result format (list-boolean).
        """
        # Eventually we could have more points but those don't have
        # any triplet information
        n_points = triplets.max() + 1
        values = np.zeros((n_points, math.comb(n_points, 2)))
        values[values == 0] = MISSING_VALUE
        labels = [(0, 0)] * (math.comb(n_points, 2))
        for i in range(triplets.shape[0]):
            a, b, c = triplets[i]
            if b > c:
                b, c = c, b
            if b == c:
                raise ValueError(
                    "Triplet information on distance of point to itself.")
            # invariant: c > b
            val = 1 if responses[i] else 0
            values[a][Questionnaire._triplet_to_pos(b, c, n_points)] = val

        for b in range(n_points):
            for c in range(b + 1, n_points):
                index = Questionnaire._triplet_to_pos(b, c, n_points)
                labels[index] = (b, c)

        if self_fill:
            return Questionnaire(values, labels).fill_self_labels()
        else:
            return Questionnaire(values, labels)

    @staticmethod
    def _triplet_to_pos(b, c, n_points) -> int:
        """
        Returns column position of the triplet information (correspondence
        of triplet and question).
        """
        return int(b * (n_points - 1) + (c - 1) - (b**2 + b) / 2)

    def fill_self_labels(self) -> Questionnaire:
        """
        Fills a questionnaire with self-information:
        A point is always closer to itself than to another point. 
        """
        v_ = self.values.copy()
        l_ = self.labels.copy()
        for (i, l) in enumerate(l_):
            v_[l[0], i] = 1
            v_[l[1], i] = 0
        return Questionnaire(v_, l_)

    @staticmethod
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
                if b > c:
                    c, b = b, c
                if (b, c) not in labels:
                    labels.append((b, c))
                    break

        return Questionnaire(xs, labels)

    def to_bool_array(self, return_responses: bool = True) -> tuple[np.ndarray, Optional[np.ndarray]]:
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

    def labels_are_ordered(self) -> bool:
        """
        Returns true, if for all labels, the index of b < c.
        """
        for l in self.labels:
            if l[0] > l[1]:
                return False
        return True

    def order_labels(self) -> Questionnaire:
        """
        Returns Questionnaire equivalent to self where all labels are ordered
        (it is always b < c).
        """
        vals = self.values.copy()
        labels = self.labels.copy()
        for i, label in enumerate(labels):
            if label[0] > label[1]:
                vals_valid = vals[:, i] != MISSING_VALUE
                vals[vals_valid, i] = np.logical_not(
                    vals[vals_valid, i]).astype(int)
                labels[i] = (label[1], label[0])
        return Questionnaire(vals, labels)

    def fill_with(self, other: Questionnaire) -> Questionnaire:
        """
        Imputes missing values in this questionnaire with the other questionnaire.

        Returns the imputed questionnaire, doesn't change self.

        The return questionnaire may still have missing values, if the other
        questionnaire had missing values as well.
        """
        vals = self.values.copy()
        labels = self.labels.copy()
        label_indices_not_in_this_questionnaire = []
        labels_not_in_this_questionnaire = []
        for (i, label) in enumerate(other.labels):
            assert label[0] < label[1]
            if label not in labels:
                labels_not_in_this_questionnaire.append(label)
                label_indices_not_in_this_questionnaire.append(i)
            else:
                idx_in_vals = labels.index(label)
                missing = vals[:, idx_in_vals] == MISSING_VALUE
                vals[:, idx_in_vals][missing] = other.values[:, i][missing]

        vals = np.concatenate(
            (vals, other.values[:, label_indices_not_in_this_questionnaire]), axis=1)
        labels.extend(labels_not_in_this_questionnaire)
        return Questionnaire(vals, labels)

    def impute(self, imputation_method_name: str) -> Questionnaire:
        """
        Imputes the questionnaire with the given method.
        This is guaranteed not to change the original questionnaire. 
        The imputed questionnaire will be returned.

        Afterwards, the questionnaire is guaranteed to not have any 
        rows left with missing values. 

        The output questionnaire might have a different shape
        as the input questionnaire, as completely
        corrupted columns will be thrown out.

        INPUT:
            imputation_method: str
                The method to use for imputation. See ImputationMethod for possible
                values to use.
        OUTPUT:
            Questionnaire with imputed values.
        """
        if imputation_method_name is None:
            raise ValueError("No imputation method given (None received).")
        imputation_method = ImputationMethod(imputation_method_name)
        cleaned_values, cleaned_labels = Questionnaire._throwout_vals_labels(
            1, self.values, self.labels)
        return Questionnaire(imputation_method(cleaned_values), cleaned_labels)

    def throwout(self, threshhold: float):
        """
        Throws out column that has equal or more than threshhold% corrupted values.

        Returns a new questionnaire and doesn't change the old one.
        """
        new_vals, new_labels = Questionnaire._throwout_vals_labels(
            threshhold, self.values, self.labels)
        return Questionnaire(new_vals, new_labels)

    @staticmethod
    def _throwout_vals_labels(threshhold: float, values: np.ndarray, labels: "list[tuple]") -> "tuple[np.ndarray, list[tuple]]":
        """
        Computes new values and labels according to the procedure described in throwout.
        """
        corrupted_cols = (values == MISSING_VALUE).mean(axis=0) >= threshhold
        labels = np.array(labels)[~corrupted_cols]
        labels = [tuple(x) for x in labels]
        return values[:, ~corrupted_cols], labels


def create_log_function(verbose):
    if verbose:
        return lambda x: print(x)
    else:
        return lambda _: None


def _generate_questionnaire(distances: np.ndarray, noise: float = 0.0, density: float = 1.0, verbose: bool = True,
                            seed: Optional[int] = None, soft_threshhold: Optional[float] = None, imputation_method: Optional[str] = None, flip_noise: bool = False,
                            randomize_ties: bool = False, similarity: bool = False) -> Questionnaire:
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
                                soft_threshhold=soft_threshhold, flip_noise=flip_noise, randomize_tie=randomize_ties, similarity=similarity)
            answers[j] = answer
        questionnaire[i] = np.array(answers)

    questionnaire = Questionnaire(
        questionnaire, list(map(tuple, question_set)))
    if imputation_method is not None:
        questionnaire = questionnaire.impute(
            imputation_method_name=imputation_method)

    return questionnaire
