"""
Module for generating hierarchical models to cluster.
"""
from __future__ import annotations
from multiprocessing.sharedctypes import Value
from typing import Optional
from copy import deepcopy
import numpy as np


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


def get_primitives(elements: list[int]):
    return [HierarchyNode(set([element]), None) for element in elements]
