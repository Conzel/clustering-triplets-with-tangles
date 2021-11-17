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
from operator import itemgetter
from tqdm import tqdm


def distance_function(x, y): return np.linalg.norm(x - y)


def is_triplet(a, b, c, dist=distance_function, noise=0.0):
    """"
    Returns 1 if a is closer to b than c, 0 otherwise.
    If noise > 0, then the question is randomly answered with probability 1 - noise.
    """
    if noise > 0 and random.random() < noise:
        return random.randint(0, 1)
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
    if density > 0.1: # very rough, we could use a better metric here
        question_set = generate_k_subsets(list(range(num_datapoints)), 2)
        assert len(question_set) == max_amount_of_questions
        # Removing questions from the set of all questions
        if density < 1.0:
            actual_amount_of_questions = math.floor(max_amount_of_questions * density)
            idx = random.sample(range(max_amount_of_questions), actual_amount_of_questions)
            question_set = itemgetter(*idx)(question_set)

        return question_set

    # very sparse set, we are better off directly sampling numbers
    else:
        total_questions = math.comb(num_datapoints, 2) * density
        sampled_questions = set()
        num_sampled = 0
        while num_sampled < total_questions:
            b, c = random.randint(
                0, num_datapoints-1), random.randint(0, num_datapoints-1)
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

    def __init__(self, questionnaire: np.ndarray, labels: "list[tuple]") -> None:
        assert len(labels) == questionnaire.shape[1]
        self.values = questionnaire
        self.labels = labels


def create_log_function(verbose):
    if verbose:
        return lambda x: print(x)
    else:
        return lambda _: None


def generate_questionnaire(data: data_types.Data, noise=0.0, density=1.0, verbose=True, seed=None) -> Questionnaire:
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

    return Questionnaire(questionnaire, list(map(tuple, question_set)))