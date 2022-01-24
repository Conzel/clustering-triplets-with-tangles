use crate::cuts::{is_consistent, is_subset, CutId};
use crate::cuts::{Core, Cut, CutOrientation, CutPool, CutValue};
use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
/// Maximum number of elements in the Tangles tree
type UTreeSize = u16;

enum Side {
    Left,
    Right,
}

// Arena based implementation of trees
#[derive(Clone, PartialEq)]
pub struct TanglesTree<'a> {
    nodes: Vec<TanglesTreeNode<'a>>,
    pool: CutPool,
    next_free_idx: UTreeSize,
    agreement: u16,
}

impl<'a> std::fmt::Debug for TanglesTree<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TanglesTree {{ nodes: [")?;
        for (i, node) in self.nodes.iter().enumerate() {
            write!(f, "\n")?;
            write!(f, "{}. {:?}", node.id, node)?;
            if i != self.nodes.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(
            f,
            "], next_free_idx: {:?}, agreement: {:?} }}",
            self.next_free_idx, self.agreement
        )
    }
}

#[derive(Clone, PartialEq)]
struct TanglesTreeNode<'a> {
    left: Option<UTreeSize>,
    right: Option<UTreeSize>,
    parent: Option<UTreeSize>,
    id: UTreeSize,
    value: Option<Cut>,
    core: Cow<'a, Core>,
}

impl std::fmt::Debug for TanglesTreeNode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let left_child = match self.left {
            Some(id) => format!("{}", id),
            None => "-".to_string(),
        };
        let right_child = match self.right {
            Some(id) => format!("{}", id),
            None => "-".to_string(),
        };
        let par = match self.parent {
            Some(id) => format!("{}", id),
            None => "-".to_string(),
        };
        let val = match self.value {
            Some(cut) => format!("{:?}", cut),
            None => "-".to_string(),
        };
        write!(
            f,
            "TanglesTreeNode {{ children: ({},{}), parent: {}, id: {:?}, value: {}, core: {:?} }}",
            left_child, right_child, par, self.id, val, self.core
        )
    }
}

impl<'a> TanglesTreeNode<'a> {
    fn new(
        parent: Option<UTreeSize>,
        id: UTreeSize,
        value: Option<Cut>,
        core: Cow<'a, Core>,
    ) -> TanglesTreeNode {
        TanglesTreeNode {
            left: None,
            right: None,
            parent,
            id,
            value,
            core,
        }
    }

    fn is_splitting_node(&self) -> bool {
        self.left.is_some() && self.right.is_some()
    }

    fn is_leaf_node(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Returns the side that the child node indicated by child_id
    /// is on (or returns None);
    fn which_child(&self, child_id: UTreeSize) -> Option<Side> {
        if let Some(id) = self.left {
            if id == child_id {
                return Some(Side::Left);
            }
        }
        if let Some(id) = self.right {
            if id == child_id {
                return Some(Side::Right);
            }
        }
        return None;
    }
}

/// An iterator that traverses up from the current node (inclusive)
/// to the root node (inclusive).
struct TanglesTreeAncestorIter<'a> {
    current: Option<UTreeSize>,
    tree: &'a TanglesTree<'a>,
}

impl<'a> Iterator for TanglesTreeAncestorIter<'a> {
    type Item = &'a TanglesTreeNode<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        current.map(|id| {
            let node = self.tree.get_node_at(id);
            let parent = node.parent;
            self.current = parent;
            node
        })
    }
}

/// An iterator that traverses level wise from the subtree at the node it was initialized with.
struct TanglesLevelOrderIter<'a> {
    current: VecDeque<UTreeSize>,
    tree: &'a TanglesTree<'a>,
}

impl<'a> TanglesLevelOrderIter<'a> {
    fn new(at: UTreeSize, tree: &'a TanglesTree<'a>) -> TanglesLevelOrderIter<'a> {
        let mut current = VecDeque::new();
        current.push_back(at);
        TanglesLevelOrderIter { current, tree }
    }
}

impl<'a> Iterator for TanglesLevelOrderIter<'a> {
    type Item = &'a TanglesTreeNode<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_empty() {
            return None;
        }
        let next_idx = self.current.pop_front().unwrap();
        let next_node = self.tree.get_node_at(next_idx);
        if let Some(left_id) = next_node.left {
            self.current.push_back(left_id);
        }
        if let Some(right_id) = next_node.right {
            self.current.push_back(right_id);
        }
        return Some(next_node);
    }
}

impl<'a> TanglesTree<'a> {
    fn new(cuts: Vec<CutValue>, agreement: u16) -> TanglesTree<'a> {
        let root_node = TanglesTreeNode::new(None, 0, None, Cow::Owned(Vec::new()));
        TanglesTree {
            nodes: vec![root_node],
            pool: CutPool::new(cuts),
            next_free_idx: 1,
            agreement,
        }
    }

    fn get_root_idx(&'a self) -> UTreeSize {
        if self.nodes.len() > 0 {
            return 0;
        } else {
            // This shouldn't happen, as trees at least contain the root node
            panic!("TanglesTree is empty");
        }
    }

    fn contract_tree(&self) {
        todo!()
    }

    fn ancestors_iter(&self, id: UTreeSize) -> TanglesTreeAncestorIter<'_> {
        assert!(id < self.nodes.len() as UTreeSize);
        TanglesTreeAncestorIter {
            current: Some(id),
            tree: self,
        }
    }

    fn level_wise_iter(&self, id: UTreeSize) -> TanglesLevelOrderIter<'_> {
        TanglesLevelOrderIter::new(id, self)
    }

    /// Removes all paths of length less than min_path_length (from leaf
    /// to the closest splitting node / root).
    /// See Klepper et al. Algorithm 5, also p. 28, II.2
    ///
    /// We do a lazy variation of pruning, where we only remove the
    /// connections of the node, but not the nodes itself from the vector
    /// (as this would require changing all other indices). This causes the tree
    /// to not shrink in size from pruning (this might be the cause of a memory leak).
    /// In the future, we might want to provide a cleanup function, that removes
    /// unconnected nodes.
    fn prune(&mut self, min_path_length: UTreeSize) {
        let mut paths_to_parent = vec![0; self.num_nodes() as usize];

        // stores necessary changes to the tree and does them in bulk
        // (this pleases the borrow checker)
        let mut to_remove: Vec<(UTreeSize, Side)> = Vec::new();

        for current_node in self.level_wise_iter(self.get_root_idx()) {
            if current_node.is_splitting_node() || current_node.id == 0 {
                // Splitting nodes reset index.
                paths_to_parent[current_node.id as usize] = 0;
            } else {
                // Height increase as we go down the tree
                let parent_idx = current_node.parent.unwrap();
                paths_to_parent[current_node.id as usize] =
                    paths_to_parent[parent_idx as usize] + 1;
            }
            // Leaf + path too short. prune
            if current_node.is_leaf_node()
                && paths_to_parent[current_node.id as usize] <= min_path_length
            {
                for ancestor in self.ancestors_iter(current_node.id) {
                    if let Some(side) = ancestor.which_child(current_node.id) {
                        to_remove.push((ancestor.id, side));
                    }
                    if ancestor.is_splitting_node() || ancestor.id == 0 {
                        break;
                    }
                }
            }
        }

        // Cleanup: Removing the child nodes
        for (idx, side) in to_remove {
            self.remove_child_at(idx, side);
        }
    }

    /// Removes the child at the index from the tree.
    /// ! This does a lazy removal, not freeing the memory of the node.
    fn remove_child_at(&mut self, at: UTreeSize, side: Side) {
        let node = &mut self.nodes[at as usize];
        match side {
            Side::Left => {
                node.left = None;
            }
            Side::Right => {
                node.right = None;
            }
        }
    }

    /// Inserts a node at the node indicated by the at argument.
    /// The node is inserted as left or right children, depending on
    /// the side argument.
    /// The cut argument denotes the cut that the node is associated with.
    /// The core argument indicates which core the new node should hold.
    fn insert_node(
        &mut self,
        at: UTreeSize,
        side: Side,
        cut: Cut,
        core: Cow<'a, Core>,
    ) -> UTreeSize {
        let new_node_idx = self.next_free_idx;
        let new_node = TanglesTreeNode::new(Some(at), new_node_idx, Some(cut), core);
        match side {
            Side::Left => {
                assert!(cut.1 == CutOrientation::Normal);
                self.nodes[at as usize].left = Some(new_node_idx);
            }
            Side::Right => {
                assert!(cut.1 == CutOrientation::Inverted);
                self.nodes[at as usize].right = Some(new_node_idx);
            }
        }
        self.nodes.push(new_node);
        self.next_free_idx += 1;

        return new_node_idx;
    }

    fn get_node_at(&self, at: UTreeSize) -> &TanglesTreeNode<'a> {
        return &self.nodes[at as usize];
    }

    pub fn num_nodes(&self) -> UTreeSize {
        return self.nodes.len() as UTreeSize;
    }

    /// Contraction algorithm as described in Klepper et al. Algorithm 5, p. 29,
    /// "Contracting the tree."
    fn contract(&mut self) {
        todo!()
    }

    // Closely following the definition given in Klepper et al. p.29, eq. 8
    fn distinguishing_cuts(&self, at: UTreeSize) -> Vec<CutId> {
        assert!(self.get_node_at(at).is_splitting_node());
        let mut cuts = Vec::new();
        // A cut is distinguishing when:
        // - it is always oriented in the left and right subtree to the same direction
        // - it is oriented in the left and right subtree in a different fashion
        // This definition is a bit different than in eq. 8, but I believe
        // it to be equivalent, easier to implement and less convoluted.
        //
        // Thus, we want to build cut maps in the following way, per subtree:
        // - For each cut, save it's first appearance in the map under the id
        // - For later appearances, we check if the cut has been consistently
        //   oriented (orientation is the same as the saved one)
        //
        // In the end, compare for each cut in both subtrees if orientations differ
        let node = self.get_node_at(at);
        let left_map = self.consistently_oriented_cuts(node.left.unwrap());
        let right_map = self.consistently_oriented_cuts(node.right.unwrap());
        for (key, left_cut) in left_map {
            if let Some(right_cut) = right_map.get(&key) {
                if left_cut.is_some()
                    && right_cut.is_some()
                    && left_cut.unwrap().1 != right_cut.unwrap().1
                {
                    cuts.push(key);
                }
            }
        }
        return cuts;
    }

    /// Returns all cuts that are always oriented in the same direction in the
    /// subtree with root at.
    /// Cuts that have been found, but are not consistently oriented (e.g. normal
    /// once and inverted the other time) are set to none in the returned map.
    fn consistently_oriented_cuts(&self, at: UTreeSize) -> HashMap<CutId, Option<Cut>> {
        let mut map: HashMap<CutId, Option<Cut>> = HashMap::new();
        for node in self.level_wise_iter(at) {
            if let Some(cut) = node.value {
                let new_map_val = match map.get(&cut.0) {
                    // We have seen this cut already and check if it is consistent
                    Some(Some(saved_cut)) => {
                        if saved_cut.1 == cut.1 {
                            Some(cut)
                        } else {
                            None
                        }
                    }
                    // This cut has been seen already and is inconsistent
                    Some(None) => None,
                    // This cut has not been seen before
                    None => Some(cut),
                };
                map.insert(cut.0, new_map_val);
            }
        }
        return map;
    }

    /// Returns a set of all the cuts in this tangle
    fn collect_cuts(&self, at: UTreeSize) -> Vec<Cut> {
        let mut cuts = Vec::new();
        for node in self.ancestors_iter(at) {
            if let Some(cut) = node.value {
                cuts.push(cut);
            }
        }
        cuts
    }

    fn insert_node_if_consistent(
        &mut self,
        node: &u16,
        current_cut_id: u16,
        orientation: CutOrientation,
    ) -> Option<UTreeSize> {
        let side = match orientation {
            CutOrientation::Normal => Side::Left,
            CutOrientation::Inverted => Side::Right,
        };
        let candidate_cut = Cut(current_cut_id, orientation);
        if self.consistent(*node, candidate_cut) {
            let old_core = &self.get_node_at(*node).core;
            let new_core = new_core(&self.pool, old_core, candidate_cut);
            let new_node_id = self.insert_node(*node, side, candidate_cut, new_core);
            Some(new_node_id)
        } else {
            None
        }
    }

    /// Checks if the at the index given via the at argument is consistent
    /// with the cut that is indicated via the cut_id argument at the given orientation.
    ///
    /// Returns true if the cut is consistent.
    fn consistent(&self, at: UTreeSize, cut: Cut) -> bool {
        let current_core = self.get_node_at(at).core.as_ref();
        return is_consistent(cut, current_core, &self.pool, self.agreement);
    }

    pub fn pretty_print(&self) {
        let root_node = self.get_node_at(self.get_root_idx());
        let mut nodes_at_i = vec![root_node];
        let mut i = 0;
        while nodes_at_i.len() > 0 {
            let mut children = Vec::new();
            for node in nodes_at_i {
                if let Some(cut) = node.value {
                    let orientation = cut.1;
                    match orientation {
                        CutOrientation::Normal => print!(" c{} ", i),
                        CutOrientation::Inverted => print!("-c{} ", i),
                    }
                }
                if let Some(left_idx) = node.left {
                    children.push(self.get_node_at(left_idx));
                }
                if let Some(right_idx) = node.right {
                    children.push(self.get_node_at(right_idx));
                }
            }
            println!("");
            nodes_at_i = children;
            i += 1;
        }
    }
}

/// Returns the core of the node at the given index, with the cut at cut_id added.
/// Doesn't check for consistency.
fn new_core<'b>(pool: &CutPool, core: &Core, cut: Cut) -> Cow<'b, Core> {
    let cut_value = pool.cut_value(cut);
    let candidate_cut = cut_value.as_ref();
    let mut indices_to_remove = Vec::new();

    for i in (0..core.len()).rev() {
        let core_cut = &pool.cut_value(core[i]);
        if is_subset(core_cut, candidate_cut) {
            // TODO: Fix the lifetimes so we can return a borrowed cow here
            return Cow::Owned(core.to_owned());
        } else if is_subset(candidate_cut, core_cut) {
            indices_to_remove.push(i);
        }
    }

    let mut new_core = core.to_owned();
    for j in indices_to_remove {
        // Swap remove here is safe, as indices are added to the vec in
        // decreasingly sorted order.
        // Assume we remove indices 5, 2 and 0. We proceed like this:
        //    [2] [4] [7] [1] [3] [8]
        // -> [2] [4] [7] [1] [3] [ ]
        //                         ^
        // -> [2] [4] [3] [1] [ ] [ ]
        //             ^
        // -> [1] [4] [7] [ ] [ ] [ ]
        //     ^
        // Bonus: Swap remove runs in O(1) :)
        new_core.swap_remove(j);
    }
    new_core.push(cut);
    return Cow::Owned(new_core);
}

/// Algorithm 1 in the paper of Klepper et al.
/// We assume that the set of cuts is already sorted by the cut cost
pub fn tangle_search_tree<'a>(cuts: Vec<CutValue>, agreement: u16) -> TanglesTree<'a> {
    let mut tree = TanglesTree::new(cuts, agreement);
    let num_cuts = tree.pool.len();

    let mut nodes_on_layer_i = vec![tree.get_root_idx()];
    let mut current_cut_id: u16 = 0;

    while nodes_on_layer_i.len() > 0 && (current_cut_id as usize) < num_cuts {
        let mut nodes_on_next_layer: Vec<UTreeSize> = Vec::new();

        for node in &nodes_on_layer_i {
            // check consistency for both sides and add nodes in the case
            let left_node_idx =
                tree.insert_node_if_consistent(node, current_cut_id, CutOrientation::Normal);
            let right_node_idx =
                tree.insert_node_if_consistent(node, current_cut_id, CutOrientation::Inverted);

            // Adding possible node idx to the added nodes
            if let Some(left_node_idx) = left_node_idx {
                nodes_on_next_layer.push(left_node_idx);
            }
            if let Some(right_node_idx) = right_node_idx {
                nodes_on_next_layer.push(right_node_idx);
            }
        }
        // Done with all nodes on this layer, set nodes on new layer
        current_cut_id += 1;
        nodes_on_layer_i = nodes_on_next_layer;
    }
    return tree;
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use bitvec::bits;
    use bitvec::prelude::{BitVec, Lsb0};

    use crate::cuts::intersection;

    use super::*;

    fn sample_cut_pool() -> Vec<CutValue> {
        let cut_1 = bits![1, 1, 1, 0, 0, 0, 0, 0, 0];
        let cut_2 = bits![0, 0, 0, 1, 1, 1, 0, 0, 0];
        let cut_3 = bits![0, 0, 0, 0, 0, 0, 1, 1, 1];
        let cuts = vec![cut_1.into(), cut_2.into(), cut_3.into()];
        return cuts;
    }

    /// Returns a dummy tree (core is empty and cuts have no meaning).
    /// For tests on tree structure.
    /// Tree so far:
    ///       0
    ///     1   2
    ///    3 4
    ///   5
    fn sample_tree() -> TanglesTree<'static> {
        let pool = sample_cut_pool();
        let mut tree = TanglesTree::new(pool.clone(), 3);
        let core = vec![];
        tree.insert_node(
            0,
            Side::Left,
            Cut(0, CutOrientation::Normal),
            Cow::Owned(core.clone()),
        );
        tree.insert_node(
            0,
            Side::Right,
            Cut(0, CutOrientation::Inverted),
            Cow::Owned(core.clone()),
        );
        tree.insert_node(
            1,
            Side::Left,
            Cut(1, CutOrientation::Normal),
            Cow::Owned(core.clone()),
        );
        tree.insert_node(
            1,
            Side::Right,
            Cut(1, CutOrientation::Inverted),
            Cow::Owned(core.clone()),
        );
        tree.insert_node(
            3,
            Side::Left,
            Cut(2, CutOrientation::Normal),
            Cow::Owned(core.clone()),
        );
        return tree;
    }

    #[test]
    fn test_ancestors_iterator() {
        let cuts = sample_cut_pool();
        let tree = tangle_search_tree(cuts, 3);
        let mut ancestors_iter = tree.ancestors_iter(5);

        let node = ancestors_iter.next();
        assert_eq!(node.unwrap().id, 5);
        let node = ancestors_iter.next();
        assert_eq!(node.unwrap().id, 2);
        let node = ancestors_iter.next();
        assert_eq!(node.unwrap().id, 0);
        let node = ancestors_iter.next();
        assert_eq!(node, None);
    }

    #[test]
    fn test_tangle_tree_insert_left() {
        let pool = sample_cut_pool();
        let mut tree = TanglesTree::new(pool.clone(), 3);
        let cut_1 = Cut(0, CutOrientation::Normal);
        let core_1 = vec![cut_1];

        let inserted_node = TanglesTreeNode::new(Some(0), 1, Some(cut_1), Cow::Borrowed(&core_1));
        tree.insert_node(0, Side::Left, cut_1, Cow::Borrowed(&core_1));
        assert_eq!(tree.get_node_at(1), &inserted_node);
    }

    #[test]
    fn test_tangle_tree_insert_right() {
        let pool = sample_cut_pool();
        let mut tree = TanglesTree::new(pool.clone(), 3);
        let cut_1_inv = Cut(0, CutOrientation::Inverted);
        let core_1 = vec![cut_1_inv];

        let inserted_node =
            TanglesTreeNode::new(Some(0), 1, Some(cut_1_inv), Cow::Borrowed(&core_1));
        tree.insert_node(0, Side::Right, cut_1_inv, Cow::Borrowed(&core_1));
        assert_eq!(tree.get_node_at(1), &inserted_node);
    }

    #[test]
    fn test_insert_consistent_node() {
        let pool = sample_cut_pool();
        let mut tree = TanglesTree::new(pool.clone(), 3);
        let cut_1 = Cut(0, CutOrientation::Normal);
        let cut_2_inv = Cut(1, CutOrientation::Inverted);
        let cut_3_inv = Cut(2, CutOrientation::Inverted);

        // It should be allowed to insert all these nodes, as they are still consistent
        tree.insert_node_if_consistent(&0, 0, CutOrientation::Normal);
        let mut tree_after_first = tree.clone();
        tree.insert_node_if_consistent(&1, 1, CutOrientation::Inverted);
        let mut tree_after_second = tree.clone();
        tree.insert_node_if_consistent(&2, 2, CutOrientation::Inverted);
        let core_1 = vec![cut_1];

        let mut first_inserted_node =
            TanglesTreeNode::new(Some(0), 1, Some(cut_1), Cow::Borrowed(&core_1));
        first_inserted_node.right = Some(2);
        let mut second_inserted_node =
            TanglesTreeNode::new(Some(1), 2, Some(cut_2_inv), Cow::Borrowed(&core_1));
        second_inserted_node.right = Some(3);
        let last_inserted_node =
            TanglesTreeNode::new(Some(2), 3, Some(cut_3_inv), Cow::Borrowed(&core_1));
        assert_eq!(tree.get_node_at(1), &first_inserted_node);
        assert_eq!(tree.get_node_at(2), &second_inserted_node);
        assert_eq!(tree.get_node_at(3), &last_inserted_node);

        // checking inconsistent adds
        let tree_inconsistent_add_first = tree_after_first.clone();
        let tree_inconsistent_add_second = tree_after_second.clone();
        tree_after_first.insert_node_if_consistent(&1, 1, CutOrientation::Normal);
        tree_after_second.insert_node_if_consistent(&2, 2, CutOrientation::Normal);
        assert_eq!(tree_inconsistent_add_first, tree_after_first);
        assert_eq!(tree_inconsistent_add_second, tree_after_second);
    }

    #[test]
    fn test_root_is_zero() {
        let tree = TanglesTree::new(sample_cut_pool(), 3);
        assert_eq!(tree.get_root_idx(), 0);
    }

    #[test]
    fn test_subset() {
        let a: BitVec = bits![1, 1, 1, 0, 0, 0, 0, 0, 0].into();
        let b: BitVec = bits![1, 1, 1, 0, 0, 0, 0, 0, 0].into();
        let c: BitVec = bits![1, 1, 1, 1, 0, 0, 0, 0, 0].into();
        let d: BitVec = bits![1, 1, 0, 0, 1, 0, 0, 0, 0].into();
        assert!(is_subset(&a, &b));
        assert!(is_subset(&b, &a));
        assert!(is_subset(&a, &c));
        assert!(!is_subset(&c, &a));
        assert!(!is_subset(&a, &d));
        assert!(!is_subset(&b, &d));
        assert!(!is_subset(&c, &d));
        assert!(is_subset(&d, &d));
    }

    #[test]
    fn test_intersection() {
        let a: BitVec = bits![1, 1, 1, 0, 0, 0, 0, 0, 0].into();
        let b: BitVec = bits![1, 1, 1, 0, 0, 0, 0, 0, 0].into();
        let c: BitVec = bits![1, 1, 0, 1, 0, 0, 0, 0, 0].into();
        let d: BitVec = bits![1, 1, 0, 0, 0, 0, 0, 0, 0].into();
        assert_eq!(intersection(&a, &b), a);
        assert_eq!(intersection(&a, &c), d);
    }

    #[test]
    fn test_consistency() {
        let pool = CutPool::new(sample_cut_pool());
        let cut_1 = Cut(0, CutOrientation::Normal);
        let cut_2 = Cut(1, CutOrientation::Normal);
        let cut_2_inv = Cut(1, CutOrientation::Inverted);
        let cut_3 = Cut(2, CutOrientation::Normal);
        let cut_3_inv = Cut(2, CutOrientation::Inverted);

        let core_1 = vec![cut_1, cut_1];

        assert!(is_consistent(cut_2_inv, &core_1, &pool, 3));
        assert!(!is_consistent(cut_2, &core_1, &pool, 3));
        assert!(is_consistent(cut_1, &core_1, &pool, 3));
        assert!(!is_consistent(cut_1, &core_1, &pool, 5));

        let core_2 = vec![cut_1, cut_2_inv];
        assert!(is_consistent(cut_3_inv, &core_2, &pool, 3));
        assert!(!is_consistent(cut_3, &core_2, &pool, 3));
    }

    #[test]
    fn test_prune() {
        let pool = sample_cut_pool();
        let mut tree = sample_tree();
        tree.prune(1);
        // After prune:
        //       0
        //     1
        //    3
        //   5
        assert_eq!(tree.get_node_at(0).right, None);
        assert_eq!(tree.get_node_at(0).left, Some(1));
        assert_eq!(tree.get_node_at(1).left, Some(3));
        assert_eq!(tree.get_node_at(1).right, None);
        assert_eq!(tree.get_node_at(3).left, Some(5));
        tree.prune(5);
        // After prune:
        //       0
        //     1
        //    3
        //
        assert_eq!(tree.get_node_at(0).left, Some(1));
        assert_eq!(tree.get_node_at(1).left, Some(3));
        assert_eq!(tree.get_node_at(1).right, None);
        assert_eq!(tree.get_node_at(3).left, None);
        assert_eq!(tree.get_node_at(3).right, None);
    }

    #[test]
    fn test_distinguishing_cuts() {
        let mut tree = sample_tree();
        let cuts = tree.distinguishing_cuts(0);
        assert_eq!(cuts, vec![0]);
        let cuts = tree.distinguishing_cuts(1);
        assert_eq!(cuts, vec![1]);
        tree.insert_node(
            2,
            Side::Right,
            Cut(1, CutOrientation::Inverted),
            Cow::Owned(vec![]),
        );
        tree.insert_node(
            6,
            Side::Right,
            Cut(2, CutOrientation::Inverted),
            Cow::Owned(vec![]),
        );
        let mut cuts = tree.distinguishing_cuts(0);
        cuts.sort();
        assert_eq!(cuts, vec![0, 2]);
    }

    #[test]
    fn test_new_cores() {
        let cut_value_1 = bits![1, 1, 1, 1, 0, 0, 0, 0, 0].into();
        let cut_value_2 = bits![1, 1, 1, 0, 0, 0, 0, 0, 0].into();
        let cut_value_3 = bits![0, 0, 0, 0, 0, 0, 1, 1, 1].into();
        let cut_value_4 = bits![0, 0, 0, 1, 0, 0, 0, 0, 0].into();

        let cuts = vec![cut_value_1, cut_value_2, cut_value_3, cut_value_4];
        let pool = CutPool::new(cuts);
        let cut_1 = Cut(0, CutOrientation::Normal);
        let cut_2 = Cut(1, CutOrientation::Normal);
        let cut_3 = Cut(2, CutOrientation::Normal);
        let cut_4 = Cut(3, CutOrientation::Normal);

        let core_1 = vec![cut_1];
        let core_2 = vec![cut_2];
        let core_1_3 = vec![cut_1, cut_3];
        let core_3_2 = vec![cut_3, cut_2];
        let core_2_3_4 = vec![cut_2, cut_3, cut_4];

        assert_eq!(new_core(&pool, &core_1, cut_1).into_owned(), core_1);
        assert_eq!(new_core(&pool, &core_1, cut_2).into_owned(), core_2);
        assert_eq!(new_core(&pool, &core_2, cut_2).into_owned(), core_2);
        assert_eq!(new_core(&pool, &core_2, cut_1).into_owned(), core_2);
        assert_eq!(new_core(&pool, &core_1, cut_3).into_owned(), core_1_3);
        assert_eq!(new_core(&pool, &core_1_3, cut_2).into_owned(), core_3_2);
        assert_eq!(new_core(&pool, &core_2_3_4, cut_1).into_owned(), core_2_3_4);

        let cut_1_inv = Cut(0, CutOrientation::Inverted);
        let cut_2_inv = Cut(1, CutOrientation::Inverted);
        let cut_3_inv = Cut(2, CutOrientation::Inverted);

        let core_1_inv = vec![cut_1_inv];
        let core_2_inv = vec![cut_2_inv];
        let core_1_3_inv = vec![cut_1_inv, cut_3_inv];

        assert_eq!(
            new_core(&pool, &core_1_inv, cut_1_inv).into_owned(),
            core_1_inv
        );
        assert_eq!(
            new_core(&pool, &core_1_inv, cut_2_inv).into_owned(),
            core_1_inv
        );
        assert_eq!(
            new_core(&pool, &core_2_inv, cut_2_inv).into_owned(),
            core_2_inv
        );
        assert_eq!(
            new_core(&pool, &core_2_inv, cut_1_inv).into_owned(),
            core_1_inv
        );
        assert_eq!(
            new_core(&pool, &core_1_inv, cut_3_inv).into_owned(),
            core_1_3_inv
        );
    }
}
