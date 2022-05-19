import pytest
from hierarchies import HierarchyList, aari


def get_level_2_hierarchy():
    return [[[0, 2], [1, 3]], [[4, 7], [6, 5]]]


def get_level_1_hierarchy():
    return [[0, 1], [2, 3]]


def test_depth():
    assert HierarchyList(get_level_2_hierarchy()).depth == 2
    assert HierarchyList(get_level_1_hierarchy()).depth == 1
    assert HierarchyList([0, 1]).depth == 0


def test_clusters_at():
    hier = HierarchyList(get_level_2_hierarchy())
    assert hier.clusters_at_level(0) == [[0, 2, 1, 3, 4, 7, 6, 5]]
    assert hier.clusters_at_level(1) == [[0, 2, 1, 3], [4, 7, 6, 5]]
    assert hier.clusters_at_level(2) == [[0, 2], [1, 3], [4, 7], [6, 5]]


def test_elements_flat():
    hier = HierarchyList(get_level_2_hierarchy())
    assert hier.elements == [0, 2, 1, 3, 4, 7, 6, 5]
    hier = HierarchyList(get_level_1_hierarchy())
    assert hier.elements == [0, 1, 2, 3]


def test_num_elements():
    hier = HierarchyList(get_level_2_hierarchy())
    assert hier.num_elements == 8
    hier = HierarchyList(get_level_1_hierarchy())
    assert hier.num_elements == 4


def test_raises_unbalanced_depths():
    with pytest.raises(ValueError):
        HierarchyList([[1, 2], [[3, 4], 5]])


def test_raises_non_unique_elements():
    with pytest.raises(ValueError):
        HierarchyList([[1, 2], [3, 4], [5, 6], [7, 8], [8, 9]])


def test_raises_non_integers():
    with pytest.raises(ValueError):
        HierarchyList([[1, 2, 3.0, [1, 2]]])
