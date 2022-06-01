import pytest
from hierarchies import BinaryHierarchyTree, aari


def get_level_2_hierarchy():
    return [[[0, 2], [1, 3]], [[4, 7], [6, 5]]]


def get_level_1_hierarchy():
    return [[0, 1], [2, 3]]


def test_depth():
    assert BinaryHierarchyTree(get_level_2_hierarchy()).depth == 2
    assert BinaryHierarchyTree(get_level_1_hierarchy()).depth == 1
    assert BinaryHierarchyTree([0, 1]).depth == 0


def test_clusters_at():
    hier = BinaryHierarchyTree(get_level_2_hierarchy())
    assert hier.clusters_at_level(0) == [[0, 2, 1, 3, 4, 7, 6, 5]]
    assert hier.clusters_at_level(1) == [[0, 2, 1, 3], [4, 7, 6, 5]]
    assert hier.clusters_at_level(2) == [[0, 2], [1, 3], [4, 7], [6, 5]]


def test_elements_flat():
    hier = BinaryHierarchyTree(get_level_2_hierarchy())
    assert hier.elements == [0, 2, 1, 3, 4, 7, 6, 5]
    hier = BinaryHierarchyTree(get_level_1_hierarchy())
    assert hier.elements == [0, 1, 2, 3]


def test_num_elements():
    hier = BinaryHierarchyTree(get_level_2_hierarchy())
    assert hier.num_elements == 8
    hier = BinaryHierarchyTree(get_level_1_hierarchy())
    assert hier.num_elements == 4


def test_raises_non_unique_elements():
    with pytest.raises(ValueError):
        BinaryHierarchyTree([[1, 2], [3, 4], [5, 6], [7, 8], [8, 9]])


def test_raises_non_integers():
    with pytest.raises(ValueError):
        BinaryHierarchyTree([[1, 2, 3.0, [1, 2]]])


def test_unbalanced_hierarchy():
    def test_helper(tree):
        none_node = tree.root.children[1].children[1]  # type: ignore
        value_node = tree.root.children[1].children[0]  # type: ignore
        assert none_node.children == None
        assert none_node.value == None
        assert value_node.value == [4, 5, 6]
        assert tree.clusters_at_level(3) == [[0, 1], [2, 3], [
            4, 5, 6], [], [], [], [], []]
        assert tree.clusters_at_level(2) == [[0, 1], [2, 3], [4, 5, 6], []]
        assert tree.clusters_at_level(1) == [[0, 1, 2, 3], [4, 5, 6]]
    tree1 = BinaryHierarchyTree([[[0, 1], [2, 3]], 4, 5, 6])
    tree2 = BinaryHierarchyTree([[[0, 1], [2, 3]], [4, 5, 6]])
    test_helper(tree1)
    test_helper(tree2)


def test_aari():
    assert aari(BinaryHierarchyTree(get_level_2_hierarchy()),
                BinaryHierarchyTree(get_level_2_hierarchy()), 2) == 1.0
    assert aari(BinaryHierarchyTree(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), BinaryHierarchyTree(get_level_2_hierarchy()), 2) == 0.4166666666666667  # regression test
