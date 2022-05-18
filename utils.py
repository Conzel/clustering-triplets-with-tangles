from typing import TypeVar
import numpy as np
from typing import Callable


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def zero_pad_end_like(arr: np.ndarray, other: np.ndarray) -> np.ndarray:
    """
    Pads the end of the array with zeros to the given length.
    """
    return np.pad(arr, (0, other.shape[0] - arr.shape[0]), mode="constant", constant_values=(0, 0))


def flatten(l: list) -> list:
    # sum can essentially be used as a mapReduce / flatMap ;)
    if isinstance(l[0], list):
        return sum(map(flatten, l), [])
    else:
        return l


def index_cluster_list(cluster_list: list[list[int]]) -> list:
    """
    Turns a cluster described by a nested list into a list of indices.
    Example:
    [[1,3], [2,4], [5]] -> [0, 1, 0, 1, 2]
    """
    index_dict = {}
    for i, cluster in enumerate(cluster_list):
        if not isinstance(cluster, list):
            raise ValueError("Cluster list must be a list of lists.")
        for element in cluster:
            index_dict[element] = i
    l = [None] * len(index_dict)
    for key, value in index_dict.items():
        if key >= len(l):
            raise ValueError("Indices were not contigous.")
        if l[key] is not None:
            raise ValueError("Labels are not unique.")
        l[key] = value
    return l


T = TypeVar("T")
U = TypeVar("U")


def swap(t: tuple[T, U]) -> tuple[U, T]:
    assert len(t) == 2
    return t[1], t[0]


def hierarchy_list_map(l: list, f: Callable):
    """
    Maps over a hierarchy list and turns every item a into f(a)
    """
    transformed = []
    for el in l:
        if isinstance(el, list):
            transformed.append(hierarchy_list_map(el, f))
        else:
            transformed.append(f(el))
    return transformed


def hierarchy_list_flatmap(l: list, f: Callable):
    """
    Flatmap analog to hierarchy_list_map. If l is a hierarchy
    list of element of type X, then we need to have 
    f: X -> [Y], and the function would return a hierarchy list
    with elements of type Y.
    """
    transformed = []
    for el in l:
        if isinstance(el, list):
            transformed.append(hierarchy_list_flatmap(el, f))
        else:
            transformed.extend(f(el))
    return transformed
