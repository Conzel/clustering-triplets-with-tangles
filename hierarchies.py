"""
Module for generating hierarchical models to cluster.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Union
from copy import deepcopy
from ete3 import Tree
from sklearn.metrics import adjusted_rand_score
import numpy as np
from utils import index_cluster_list


class DendrogramLike(ABC):
    """
    Interface for a class that behaves like a Dendrogram: 
    We can cut off the class at some point and get the clusters of the current level.

    We can calculate the AARI for classes that implement this interface.
    """
    @abstractmethod
    def clusters_at_level(self, level: int) -> list[list[int]]:
        pass


class BinaryClusterTree():
    """
    A simple binary tree that serves as helper structure for 
    other trees used for clustering. 
    Values are stored in the leaves and represented by lists of integers.


    """
    class Node:
        """The class that we use to represent the nodes of the tree with. """

        def __init__(self, value: Optional[list[int]], parent: Optional[BinaryClusterTree.Node], children: Optional[list[BinaryClusterTree.Node]]) -> None:
            self.value = value
            self.children: Union[list[BinaryClusterTree.Node], None] = children
            self.parent = parent

        def __str__(self) -> str:
            if self.value is not None or self.children is None:
                return str(self.value)
            else:
                return "[" + ",".join([str(child) for child in self.children]) + "]"

        def __repr__(self) -> str:
            return str(self)

        def depth(self):
            """Returns 1 + maximum depth of the subtrees that this node contains."""
            if self.children is None or len(self.children) == 0:
                return 0
            else:
                return 1 + max([child.depth() for child in self.children])

        def elements_flat(self) -> list[int]:
            """Returns a list of all the elements that this node and its children hold."""
            def helper(node):
                res = []
                if node.children is None or node.children == []:
                    if node.value is None:
                        return []
                    else:
                        return node.value
                for child in node.children:
                    child_flat = helper(child)
                    res.extend(child_flat)
                return res
            return helper(self)

    def __init__(self, tree_list: list):
        """
        Builds the tree from a nested list
        Args: 
            tree_list: Nested list that represents the tree.
            Example: [[[1,2], [3,4]], 5,6] is parsed to the following tree:

                | -- [5,6]
                1   
                |     | -- [1,2]
                | -- 
                      | -- [3,4]
        """
        self.root = self.Node(None, None, None)
        self.root.children = self._build_tree(tree_list, self.root)
        self._recalculate_properties()

    @staticmethod
    def _build_tree(tree_list: list, parent: BinaryClusterTree.Node) -> list[BinaryClusterTree.Node]:
        """
        Builds up a tree from a nested list with the given parent. Returns
        list of nodes, these have to be set to the children of the given
        parent manually.
        """
        loose_elements = []
        lists = []
        for el in tree_list:
            if isinstance(el, (np.integer, int)):
                loose_elements.append(el)
            if isinstance(el, list):
                lists.append(el)
        nodes = []
        if len(lists) > 2:
            raise ValueError("Tree is not binary.")
        # add all lists as new nodes
        for l in lists:
            temp_node = BinaryClusterTree.Node(None, parent, [])
            temp_node.children = BinaryClusterTree._build_tree(l, temp_node)
            nodes.append(temp_node)
        if len(lists) == 0:
            # make parent cluster a leaf
            parent.value = loose_elements
            parent.children = None
        elif len(loose_elements) > 0:  # add loose elements as separate leaf
            nodes.append(BinaryClusterTree.Node(loose_elements, parent, None))
        return nodes

    def fill(self):
        """
        Fills the tree with nodes such that every level is completely filled.

        Mutates the tree. Node values are transferred to leftmost child 
        for nodes not on the bottom level.
        """
        target_level = self.root.depth()

        def fill_helper(node, level):
            if level == 0:
                return
            if node.children is None or node.children == []:
                value = node.value
                node.value = None
                node.children = [
                    self.Node(value, node, None), self.Node(None, node, None)]
            if len(node.children) == 1:
                node.children.append(self.Node(None, node, None))
            # fallthrough to here is by done on purpose
            for child in node.children:
                fill_helper(child, level - 1)
        fill_helper(self.root, target_level)
        self._recalculate_properties()

    def _recalculate_properties(self):
        """
        Internal method that needs to be called when the tree is mutated.
        """
        self._elements = self.root.elements_flat()
        self._num_elements = len(self.elements)
        self._depth = self.root.depth()

    @property
    def elements(self) -> list[int]:
        """All the elements contained in this tree in a flattened list."""
        return self._elements

    @property
    def num_elements(self) -> int:
        """Number of elements that this tree holds."""
        return self._num_elements

    @property
    def depth(self) -> int:
        """Maximum depth of the tree (if unbalanced)."""
        return self._depth

    def __str__(self) -> str:
        return "HierarchyTree(" + str(self.root) + ")"

    def __repr__(self) -> str:
        return str(self.root)


class BinaryHierarchyTree(BinaryClusterTree, DendrogramLike):
    """
    A tree that represents a hierarchy of clusterings in binary format.
    Each leaf is a cluster, and each internal node represents one level 
    of splitting the clusters. The tree is built from a nested list
    and filled with nodes such that every level is completely filled. 
    For details, see the fill method in the BinaryClusterTree class.

    The hierarchy is assumed to save a set of unique, contigous integers
    which can be used as labels or indices for more complex objects.
    The class enforces this constraint in the constructor.
    """

    def __init__(self, hierarchy: list):
        """
        Args:
        hierarchy: nested list of elements, with each nesting representing
            one layer. Example: [[0,1], [2,3]]
            Each element has to be an integer and unique in the whole list.
        """
        super().__init__(hierarchy)
        self.check_labels()
        self.fill()

    def clusters_at_level(self, level: int) -> list[list[int]]:
        """
        Returns all clusters at the given level. Assumes that
        the tree is filled.

        If level > depth, appends empty lists to the cluster 
        result until we have 2**level clusters.

        Examples:
        [[[0,1], [2,3]], [[4,5], [6,7]]] 
            level 0 -> [[0,1,2,3,4,5,6,7]]
            level 1 -> [[0,1,2,3], [4,5,6,7]]
            level 2 -> [[0,1], [2,3], [4,5], [6,7]]
        """
        if level < 0:
            raise ValueError(
                f"Level {level} is not in the range of the hierarchy: {self.depth}")
        if level > self.depth:
            lists_to_fill = 2**level - 2**self.depth 
            level = self.depth
        else:
            lists_to_fill = 0

        def helper(node, level):
            if level == 0:
                return [node.elements_flat()]
            return sum([helper(child, level - 1) for child in node.children], [])

        res = helper(self.root, level)
        # extending to the necessary level, if needed
        res.extend([[]] * lists_to_fill)
        return res

    def check_labels(self):
        """
        Checks that each label (element of the hierarchy) is a unique integer,
        and that the integers are contigous, starting with 0.
        """
        for element in self.elements:
            if not isinstance(element, (np.integer, int)):
                raise ValueError(f"Element {element} is not an integer.")
        if not len(np.unique(self.elements)) == len(self.elements):
            raise ValueError(f"Elements are not unique: {self.elements}")
        if not (min(self.elements) == 0 and max(self.elements) == len(self.elements) - 1):
            raise ValueError(f"Elements are not contigous: {self.elements}")


def aari(hierarchy_left: DendrogramLike, hierarchy_right: DendrogramLike, depth: int) -> float:
    """
    Returns the similarity between the two hierarchies according to the 
    Average Adjusted Rand Index (see the appendix Ghoshdastidar et al., 2019
    for a more thorough definition).
    """
    aris = []
    for l in range(1, depth + 1):
        clusters_left = index_cluster_list(hierarchy_left.clusters_at_level(l))
        clusters_right = index_cluster_list(
            hierarchy_right.clusters_at_level(l))
        aris.append(adjusted_rand_score(clusters_left, clusters_right))
    return np.mean(aris)


if __name__ == "__main__":
    b = BinaryClusterTree([[[0, 1], [2, 3]], 4, 5, 6])
    b.fill()
    b.root.elements_flat()
