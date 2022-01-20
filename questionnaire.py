"""
File containing functions and class definitions for working with triplets in questionnaire form. 
"""
import math
import random
import re
from operator import itemgetter

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.neighbors import DistanceMetric
from tqdm import tqdm

# Allows us to import tangles modules

MISSING_VALUE = -1


def distance_function(x, y):
    return np.linalg.norm(x - y)


def is_triplet(a, b, c, distances, noise=0.0):
    """"
    Returns 1 if a is closer to b than c, 0 otherwise.
    If noise > 0, then the questions answer is set to -1 with probability noise.

    distances: ndarray of shape (num_datapoints, num_datapoints),
        precomputed distances, where distances_ij = distance between point i and j
    """
    if noise > 0 and random.random() < noise:
        return -1
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

    def from_bool_array(triplets, responses):
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

    def to_bool_array(self) -> "tuple(np.ndarray, np.ndarray)":
        """
        Transforms triplet value matrix to a list representing the triplets,
        conforming to the interface in David Künstles cblearn package under
        'triplet-array'

        https://cblearn.readthedocs.io/en/latest/generated_examples/triplet_formats.html?highlight=matrix

        To Quote:

        In the array format, the constraints are encoded by the index order.

        triplet = triplets_ordered[0]
        print(f"The triplet {triplet} means, that object {triplet[0]} (1st) should be "
            f"embedded closer to object {triplet[1]} (2nd) than to object {triplet[0]} (3th).")

        Answer Array:
        triplets_boolean, answers_boolean = check_query_response(triplets_ordered, result_format='list-boolean')
        print(f"Is object {triplets_boolean[0, 0]} closer to object {triplets_boolean[0, 1]} "
            f"than to object {triplets_boolean[0, 2]}? {answers_boolean[0]}.")

        Returns a tuple where the first part is the triplet array, the second part is the answer 
        array (all as specified above).
        """
        triplets = []
        responses = self.values.flatten()
        labels_np = np.array(self.labels)
        num_questions = labels_np.shape[0]
        for a in range(self.values.shape[0]):
            a_array = np.repeat(a, num_questions).reshape(-1, 1)
            triplets.append(np.hstack((a_array, labels_np)))
        return np.concatenate(triplets), responses == 1

    def impute(self, imputation_method: str) -> "Questionnaire":
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


def generate_questionnaire(data: np.ndarray, noise=0.0, density=1.0, verbose=True, seed=None, imputation_method=None) -> Questionnaire:
    """
    Generates a questionnaire for the given data.

    Input: 
    - data: np.ndarray
      contains the data points as rows and their coordinates as columns
    - noise: float
      The percentage of noise to add to the questionnaire. 0 means all answers are truthful, 
      1 means all answers are random.
    - density: float
      The peprcentage of questions we know the answers. 1.0 means we know the answers to all questions, 
      0.0 means we don't know any of them. This amounts to the number of columns that will be contained 
      in the questionnaire.
    - seed: int

    --------------------------------------------------------------------------------------------
    Output: Questionnaire

    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    assert 0 <= noise <= 1
    assert 0 <= density <= 1

    log = create_log_function(verbose)
    num_datapoints = data.shape[0]
    log("Generating questionnaire...")

    log("Generating question set...")
    question_set = generate_question_set(num_datapoints, density)

    # Iterate over all points and answer all questions for them.
    # The questionnaire contains all answers for all questions.
    log("Filling out questionnaire...")
    questionnaire = np.zeros((num_datapoints, len(question_set)))

    metric = DistanceMetric.get_metric("euclidean")
    # cached distances for all points
    distances = metric.pairwise(data)

    for i in tqdm(range(num_datapoints), disable=not verbose):
        a = i
        answers = np.zeros(len(question_set))
        for j, question in enumerate(question_set):
            b = question[0]
            c = question[1]
            answer = is_triplet(a, b, c, distances, noise=noise)
            answers[j] = answer
        questionnaire[i] = np.array(answers)
    
    questionnaire = Questionnaire(questionnaire, list(map(tuple, question_set)))
    if imputation_method is not None:
        questionnaire = questionnaire.impute(imputation_method=imputation_method)

    return questionnaire
