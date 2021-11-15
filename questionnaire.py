"""
File containing functions and class definitions for working with triplets in questionnaire form. 
"""
# Allows us to import tangles modules
import sys
sys.path.append("./tangles")
import numpy as np
import math
import src.data_types as data_types


def distance_function(x, y): return np.linalg.norm(x - y)


def is_triplet(a, b, c, dist=distance_function):
    """"
    Returns 1 if a is closer to b than c, 0 otherwise
    """
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


def generate_questionnaire(data: data_types.Data) -> Questionnaire:
    """
    Generates a questionnaire for the given data.

    Input: Data, in which the xs field contains the data points as rows and their coordinates
    as columns
    --------------------------------------------------------------------------------------------

    """
    total_num_samples = data.xs.shape[0]
    # Generate the triplet table
    question_set = generate_k_subsets(list(range(total_num_samples)), 2)
    assert len(question_set) == math.comb(total_num_samples, 2)

    # Iterate over all points and answer all questions for them.
    # The questionnaire contains all answers for all questions.
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

    return Questionnaire(questionnaire, list(map(tuple, question_set)))
