"""
File containing functions and class definitions for working with triplets in questionnaire form. 
"""
# Allows us to import tangles modules
import sys
sys.path.append("./tangles")
import numpy as np
import math
import src.data_types as data_types
import random
import re
from sklearn.impute import KNNImputer
from enum import Enum
from operator import itemgetter
from tqdm import tqdm


def distance_function(x, y): return np.linalg.norm(x - y)


def is_triplet(a, b, c, dist=distance_function, noise=0.0):
    """"
    Returns 1 if a is closer to b than c, 0 otherwise.
    If noise > 0, then the questions answer is set to -1 with probability noise.
    """
    if noise > 0 and random.random() < noise:
        return -1
    else:
        return int(dist(a, b) <= dist(a, c))


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

    RANDOM: Fills in a random value.
    NEIGHBOURS: Fills in the value with the mean of the most common n neighbours.
    MEAN: Fills in the value with the dataset mean.
    THROWOUT: Throws out column with corrupted value (only possible for very low noise).
    """

    def __init__(self, method_name: str):
        """
        Initiates the imputation method. Additional arguments are
        given through the constructor and might be required for methods.
        Neighbours imputation f.e. needs to know how many neighbours to use.
        """
        self.method = ImputationMethod._parse_imputation(method_name)

    def _impute_random(data: np.ndarray):
        """
        Imputes missing values with a random value.
        """
        imputed_data = data.copy()
        imputed_data[imputed_data == -1] = np.random.randint(0, 2, imputed_data[imputed_data == -1].shape)
        return imputed_data

    def _impute_knn(data: np.ndarray, k: int):
        """
        Imputes missing values with the mean value of the k nearest neighbours. 
        Coinflip decides on 0.5.
        """
        imputer = KNNImputer(n_neighbors=k, missing_values=-1)
        imputed_data = imputer.fit_transform(data)
        # removing the 0.5 values with random values
        imputed_data[imputed_data == 0.5] = np.random.randint(0, 2, imputed_data[imputed_data == 0.5].shape)
        return imputed_data

    def _impute_mean(data: np.ndarray):
        """
        Imputes missing values with the mean value of the column.
        """
        raise NotImplementedError

    def _impute_throwout(data: np.ndarray):
        """
        Throws out column with corrupted value (only possible for very low noise).
        """
        raise NotImplementedError

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
        elif imputation_method.lower() == "throwout":
            return ImputationMethod._impute_throwout

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


def create_log_function(verbose):
    if verbose:
        return lambda x: print(x)
    else:
        return lambda _: None


def generate_questionnaire(data: data_types.Data, noise=0.0, imputation_method=None, density=1.0, verbose=True, seed=None) -> Questionnaire:
    """
    Generates a questionnaire for the given data.

    Input: 
    - data: Data
      xs field contains the data points as rows and their coordinates as columns
    - noise: float
      The percentage of noise to add to the questionnaire. 0 means all answers are truthful, 
      1 means all answers are random.
    - density: float
      The peprcentage of questions we know the answers. 1.0 means we know the answers to all questions, 
      0.0 means we don't know any of them. This amounts to the number of columns that will be contained 
      in the questionnaire.
    - seed: int

    --------------------------------------------------------------------------------------------

    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if noise > 0 and imputation_method is None:
        raise ValueError("No imputation method given for noisy data.")
    assert 0 <= noise <= 1
    assert 0 <= density <= 1

    log = create_log_function(verbose)
    num_datapoints = data.xs.shape[0]
    log("Generating questionnaire...")

    log("Generating question set...")
    question_set = generate_question_set(num_datapoints, density)

    # Iterate over all points and answer all questions for them.
    # The questionnaire contains all answers for all questions.
    log("Filling out questionnaire...")
    questionnaire = np.zeros((num_datapoints, len(question_set)))

    for i in tqdm(range(num_datapoints), disable=not verbose):
        a = data.xs[i]
        answers = []
        for question in question_set:
            b = data.xs[question[0]]
            c = data.xs[question[1]]
            answer = is_triplet(a, b, c, noise=noise)
            answers.append(answer)
        questionnaire[i] = np.array(answers)

    if noise > 0:
        log("Imputing missing answers in the questionnaire...")
        imputation_method(questionnaire)

    return Questionnaire(questionnaire, list(map(tuple, question_set)))
