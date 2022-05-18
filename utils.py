import numpy as np


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def zero_pad_end_like(arr: np.ndarray, other: np.ndarray) -> np.ndarray:
    """
    Pads the end of the array with zeros to the given length.
    """
    return np.pad(arr, (0, other.shape[0] - arr.shape[0]), mode="constant", constant_values=(0, 0))


def flatten(l: list[list]) -> list:
    # sum can essentially be used as a mapReduce / flatMap ;)
    return sum(l, [])
