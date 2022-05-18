"""
Module for generating hierarchical models to cluster.
"""
from __future__ import annotations
from multiprocessing.sharedctypes import Value
from typing import Optional
from copy import deepcopy
from ete3 import Tree
from sklearn.metrics import adjusted_rand_score
import numpy as np
from utils import index_cluster_list, flatten


class HierarchyNode:
    def __init__(self, value: set[int], parent: Optional[HierarchyNode]):
        self.value = value
        self.parent = parent
        self.children = []
        self.level = 0

    def __str__(self):
        return str(self.value)

    def __repr__(self) -> str:
        return str(self.value)

    def update_levels(self):
        """
        Recursively updates levels of this node and all its children. 
        """
        if self.parent is None:
            self.level = 0
        else:
            self.level = self.parent.level + 1
        for child in self.children:
            child.update_levels()

    def add_children(self, children: list[HierarchyNode]):
        for child in children:
            child.parent = self
            self.children.append(child)

    def _newick_tree(self) -> str:
        if self.children == []:
            if len(self.value) == 1:
                return str(next(iter(self.value)))
            else:
                raise ValueError(f"Leaf has not only one value: {self.value}")
        else:
            s = "(" + ",".join([child._newick_tree()
                                for child in self.children]) + ")"
            if self.parent is None:
                return s + ";"
            else:
                return s


def merge_nodes(nodes: list[HierarchyNode]) -> HierarchyNode:
    """
    Builds a new node from a list of previous nodes.
    """
    nodes = [deepcopy(node) for node in nodes]
    new_node = HierarchyNode(set.union(*[node.value for node in nodes]), None)
    new_node.add_children(nodes)
    new_node.update_levels()
    return new_node


class HierarchyTree:
    def __init__(self, root: HierarchyNode):
        self.root = root
        root.update_levels()
        assert root.parent == None
        assert root.level == 0

    def closest_ancestor_level(self, a: int, b: int) -> int:
        """
        Returns the level of the closest ancestor of both a and b.

        Note that if this number is lower, then a and b are further away
        from each other.
        """
        if not (a in self.root.value and b in self.root.value):
            raise ValueError(
                f"Tree does not contain nodes {a} and {b}. Stored labels: {self.root.value}")
        current_node = self.root
        while True:
            eligible_children = []
            for child in current_node.children:
                if a in child.value and b in child.value:
                    eligible_children.append(child)
            if len(eligible_children) == 0:
                return current_node.level
            if len(eligible_children) == 1:
                current_node = eligible_children[0]
                continue
            if len(eligible_children) > 1:
                raise ValueError("Erroneous hierarchy, hierarchies overlap.")

    def draw(self):
        newick = self.root._newick_tree()
        print(Tree(newick))


def get_primitives(elements: list[int]):
    return [HierarchyNode(set([element]), None) for element in elements]


class HierarchyList():
    """Hierarchy that is described as a nested list, like so:
    [[0,1], [2,3]].
    The hierarchy is assumed to save a set of unique, contigous integers
    which can be used as labels or indices for more complex objects.
    The class enforces this constraint in the constructor.

    Properties:
    elements: list of all elements in the hierarchy, flattened
    hierarchy: raw hierarchy as nested list.
    """

    def __init__(self, hierarchy: list) -> None:
        """
        Args:
        hierarchy: nested list of elements, with each nesting representing
            one layer. Example: [[0,1], [2,3]]
            Each element has to be an integer and unique in the whole list.
        """
        self._depth = HierarchyList._determine_depth(hierarchy)
        self.elements = HierarchyList._elements_flat(hierarchy)
        self.hierarchy = hierarchy
        self.check_labels()

    def check_labels(self):
        """
        Checks that each label (element of the hierarchy) is a unique integer,
        and that the integers are contigous, starting with 0.
        """
        for element in self.elements:
            if not isinstance(element, int):
                raise ValueError(f"Element {element} is not an integer.")
        if not len(np.unique(self.elements)) == len(self.elements):
            raise ValueError(f"Elements are not unique: {self.elements}")
        if not (min(self.elements) == 0 and max(self.elements) == len(self.elements) - 1):
            raise ValueError(f"Elements are not contigous: {self.elements}")

    @property
    def depth(self) -> int:
        """
        Returns depth of the hierarchy. All subtrees of the hierarchy 
        have the same depth.
        """
        return self._depth

    @property
    def num_elements(self) -> int:
        """
        Returns the total number of elements in the hierarchy (flattened).
        """
        return len(self.elements)

    def clusters_at_level(self, l: int) -> list[list[int]]:
        """
        Returns all clusters at level l.

        Examples:
        [[[0,1], [2,3]], [[4,5], [6,7]]] 
            level 0 -> [[0,1,2,3,4,5,6,7]]
            level 1 -> [[0,1,2,3], [4,5,6,7]]
            level 2 -> [[0,1], [2,3], [4,5], [6,7]]
        """
        if l < 0 or l > self.depth:
            raise ValueError(
                f"Level {l} is not in the range of the hierarchy: {self.depth}")

        def helper(l: int, hierarchy: list):
            if l == 0:
                return [HierarchyList._elements_flat(hierarchy)]
            else:
                return sum([(helper(l - 1, child)) for child in hierarchy], [])

        return helper(l, self.hierarchy)

    @staticmethod
    def _elements_flat(hierarchy) -> list:
        res = []
        for el in hierarchy:
            if isinstance(el, list):
                res.extend(HierarchyList._elements_flat(el))
            else:
                res.append(el)
        return res

    @staticmethod
    def _determine_depth(hierarchy) -> int:
        depths = []
        for el in hierarchy:
            if isinstance(el, list):
                depths.append(1 + HierarchyList._determine_depth(el))
            elif isinstance(el, int):
                depths.append(0)
            else:
                raise ValueError(f"Unrecognized type: {type(el)}")
        for d in depths:
            if d != depths[0]:
                raise ValueError(
                    f"Passed hierarchy {hierarchy} has different depths, currently not supported.")
        return depths[0]

    def __str__(self) -> str:
        return "Hierarchy(" + str(self.hierarchy) + ")"

    def __repr__(self) -> str:
        return self.__str__()


def aari(hierarchy_left: HierarchyList, hierarchy_right: HierarchyList) -> float:
    """
    Returns the similarity between the two hierarchies according to the 
    Average Adjusted Rand Index (see the appendix Ghoshdastidar et al., 2019
    for a more thorough definition).
    """
    if hierarchy_left.depth != hierarchy_right.depth:
        raise ValueError(
            f"Hierarchies have different depths: {hierarchy_left.depth} and {hierarchy_right.depth}")
    aris = []
    for l in range(1, hierarchy_left.depth + 1):
        clusters_left = index_cluster_list(hierarchy_left.clusters_at_level(l))
        clusters_right = index_cluster_list(
            hierarchy_right.clusters_at_level(l))
        aris.append(adjusted_rand_score(clusters_left, clusters_right))
    return np.mean(aris)
