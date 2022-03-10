from hierarchies import HierarchyTree, get_primitives, merge_nodes
from questionnaire import Questionnaire
import numpy as np


def test_building_tree():
    nodes = get_primitives([0, 1, 2, 3, 4, 5])
    nodes012 = merge_nodes([nodes[0], nodes[1], nodes[2]])
    nodes34 = merge_nodes([nodes[3], nodes[4]])
    nodes345 = merge_nodes([nodes34, nodes[5]])
    root = merge_nodes([nodes012, nodes345])
    tree = HierarchyTree(root)
    assert tree.closest_ancestor_level(0, 5) == 0
    assert tree.closest_ancestor_level(5, 5) == 2
    assert tree.closest_ancestor_level(1, 5) == 0
    assert tree.closest_ancestor_level(0, 0) == 2
    assert tree.closest_ancestor_level(3, 4) == 2
    assert tree.closest_ancestor_level(3, 5) == 1


def test_questionnaire_from_hierarchy():
    nodes = get_primitives([0, 1, 2])
    nodes01 = merge_nodes([nodes[0], nodes[1]])
    root = merge_nodes([nodes01, nodes[2]])
    tree = HierarchyTree(root)

    data = [0, 1, 2]
    qv = np.zeros((3, 3))
    for i in range(10000):
        qv += Questionnaire.from_hierarchy(tree,
                                           np.array(data), verbose=False, randomize_ties=True).values
    assert np.abs(
        qv / 10000 - np.array([[1, 1, 1], [0, 1, 1], [0.5, 0, 0]])).sum() < 0.01
