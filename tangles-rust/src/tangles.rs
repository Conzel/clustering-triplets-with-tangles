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

#[derive(Clone)]
pub struct ContractedTanglesTree {
    nodes: Vec<ContractedTanglesNode>,
    next_free_idx: UTreeSize,
    costs: Vec<f64>,
    pool: CutPool,
}

#[derive(Clone)]
pub struct ContractedTanglesNode {
    pub distinguishing_cuts: Vec<Cut>,
    pub id: UTreeSize,
    pub parent: Option<UTreeSize>,
    pub left: Option<UTreeSize>,
    pub right: Option<UTreeSize>,
    pub cost_normalizer: f64,
}

impl ContractedTanglesNode {
    pub fn new(
        distinguishing_cuts: Vec<Cut>,
        id: UTreeSize,
        parent: Option<UTreeSize>,
        costs: &Vec<f64>,
    ) -> ContractedTanglesNode {
        let cost_normalizer = distinguishing_cuts
            .iter()
            .map(|cut| costs[cut.0 as usize].clone())
            .sum();
        ContractedTanglesNode {
            distinguishing_cuts,
            id,
            parent,
            left: None,
            right: None,
            cost_normalizer,
        }
    }
}

impl std::fmt::Debug for ContractedTanglesNode {
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
        write!(
            f,
            "TanglesTreeNode {{ children: ({},{}), parent: {}, id: {:?}, value: {:?}}}",
            left_child, right_child, par, self.id, self.distinguishing_cuts
        )
    }
}

impl ContractedTanglesTree {
    /// Initializes a new tangles tree. Costs are a vector of costs for each CutId
    /// in the pool of all_cuts.
    /// For the distinguishing cuts, it holds, that this _always_ describes the orientation
    /// in the right subtree.
    pub fn new(
        costs: Vec<f64>,
        root_distinguishing_cuts: Vec<Cut>,
        pool: CutPool,
    ) -> ContractedTanglesTree {
        assert!(costs.len() == pool.len());
        let node = ContractedTanglesNode::new(root_distinguishing_cuts, 0, None, &costs);
        ContractedTanglesTree {
            nodes: vec![node],
            next_free_idx: 1,
            costs,
            pool,
        }
    }

    /// If we want to add, we always have to give two children,
    /// as a contracted tree only contains leaves and splitting nodes.
    ///
    /// For the distinguishing cuts, it holds, that this _always_ describes the orientation
    /// in the right subtree.
    fn insert_children(
        &mut self,
        at: UTreeSize,
        left_distinguishing_cuts: Vec<Cut>,
        right_distinguishing_cuts: Vec<Cut>,
    ) {
        self.insert_node(at, left_distinguishing_cuts, Side::Left);
        self.insert_node(at, right_distinguishing_cuts, Side::Right);
    }

    fn insert_node(
        &mut self,
        at: UTreeSize,
        distinguishing_cuts: Vec<Cut>,
        side: Side,
    ) -> UTreeSize {
        let node = ContractedTanglesNode::new(
            distinguishing_cuts,
            self.next_free_idx,
            Some(at),
            &self.costs,
        );
        let node_id = node.id;
        match side {
            Side::Left => {
                self.nodes[at as usize].left = Some(node.id);
            }
            Side::Right => {
                self.nodes[at as usize].right = Some(node.id);
            }
        }
        self.nodes.push(node);
        self.next_free_idx += 1;
        node_id
    }

    /// v refers to an element q in the bipartitions described by the cuts.
    /// If c is a cut value, v is the index of q at position
    /// c[v].
    /// Returns a vector of probabilities that sums to one, where
    /// q belongs to the i-th tangle with probability vec[i].
    pub fn probabilities(&self, v: u16) -> Vec<f64> {
        self.probabilities_recursive(0, v, 1.0)
    }

    fn probabilities_recursive(&self, at: UTreeSize, v: u16, p: f64) -> Vec<f64> {
        if self.is_leaf(at) {
            vec![p]
        } else {
            // Must be a splitter node
            let node = &self.nodes[at as usize];
            let (left_split_probability, right_split_probability) =
                self.splitting_probabilities(at, v).unwrap();
            let mut left_probabilities =
                self.probabilities_recursive(node.left.unwrap(), v, p * left_split_probability);
            let mut right_probabilities =
                self.probabilities_recursive(node.right.unwrap(), v, p * right_split_probability);
            left_probabilities.append(&mut right_probabilities);
            left_probabilities
        }
    }

    pub fn is_leaf(&self, at: UTreeSize) -> bool {
        self.nodes[at as usize].left.is_none() && self.nodes[at as usize].right.is_none()
    }

    /// Returns the probability that the object q referenced by v
    /// takes the left resp. right branch of the tree at position at.
    pub fn splitting_probabilities(&self, at: UTreeSize, v: u16) -> Option<(f64, f64)> {
        if self.is_leaf(at) {
            return None;
        }
        let node = &self.nodes[at as usize];
        let right_probability = node
            .distinguishing_cuts
            .iter()
            .filter(|c| self.pool.is_in(**c, v))
            .map(|c| self.costs[c.0 as usize])
            .sum::<f64>()
            / node.cost_normalizer;
        return Some((1.0 - right_probability, right_probability));
    }
}

impl std::fmt::Debug for ContractedTanglesTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContractedTree {{ nodes: [")?;
        for (i, node) in self.nodes.iter().enumerate() {
            write!(f, "\n")?;
            write!(f, "{}. {:?}", node.id, node)?;
            if i != self.nodes.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "], next_free_idx: {:?}}}", self.next_free_idx)
    }
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

    /// Returns the index of the closest splitting node
    /// in the subtree at the given index, or None if there are None.
    fn find_next_splitting_node(&self, at: UTreeSize) -> Option<UTreeSize> {
        for node in self.level_wise_iter(at) {
            if node.is_splitting_node() {
                return Some(node.id);
            }
        }
        return None;
    }

    /// Searches for the distinguishing cuts of the next splitter node.
    /// Returns empty vec if no splitter nodes are in the subtree.
    fn next_distinguishing_cuts(&self, at: UTreeSize) -> Vec<Cut> {
        if let Some(splitter) = self.find_next_splitting_node(at) {
            self.distinguishing_cuts(splitter)
        } else {
            Vec::new()
        }
    }

    /// Contracts the tree as described in Algorithm 5.
    /// One might wish to call prune beforehand to remove
    /// noisy node paths.
    pub fn contract_tree(&self, costs: Vec<f64>) -> ContractedTanglesTree {
        let mut contracted_tree =
            ContractedTanglesTree::new(costs, self.distinguishing_cuts(0), self.pool.clone());
        // Every entry consists of
        //  (current_index,
        //   side of the parent subtree this node belongs to,
        //   index of the last parent node)
        let root_node = self.get_node_at(0);
        if root_node.is_leaf_node() {
            return contracted_tree;
        }
        let mut nodes_stack = vec![
            (root_node.left.unwrap(), Side::Left, 0),
            (root_node.right.unwrap(), Side::Right, 0),
        ];

        while nodes_stack.len() > 0 {
            let (current_node_idx, parent_side, last_parent) = nodes_stack.pop().unwrap();
            let current_node = self.get_node_at(current_node_idx);

            if current_node.is_splitting_node() {
                let new_node_id = contracted_tree.insert_node(
                    last_parent,
                    self.distinguishing_cuts(current_node.id),
                    parent_side,
                );
                nodes_stack.push((current_node.left.unwrap(), Side::Left, new_node_id));
                nodes_stack.push((current_node.right.unwrap(), Side::Right, new_node_id));
            } else if current_node.is_leaf_node() {
                contracted_tree.insert_node(
                    last_parent,
                    self.distinguishing_cuts(current_node.id),
                    parent_side,
                );
            } else {
                if let Some(left_id) = current_node.left {
                    nodes_stack.push((left_id, parent_side, last_parent));
                } else if let Some(right_id) = current_node.right {
                    nodes_stack.push((right_id, parent_side, last_parent));
                }
            }
        }
        return contracted_tree;
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
    pub fn prune(&mut self, min_path_length: UTreeSize) {
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

    /// Closely following the definition given in Klepper et al. p.29, eq. 8
    fn distinguishing_cuts(&self, at: UTreeSize) -> Vec<Cut> {
        // We could probably be a bit more efficient by using the procedure
        // described in the pseudo-code (propagating information up by the leaves).
        if self.get_node_at(at).is_leaf_node() {
            return vec![];
        }
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
        // We always push back the orientation of the cut in the right subtree.
        let node = self.get_node_at(at);
        let left_map = self.consistently_oriented_cuts(node.left.unwrap());
        let right_map = self.consistently_oriented_cuts(node.right.unwrap());
        for (key, left_cut) in left_map {
            if let Some(right_cut) = right_map.get(&key) {
                if left_cut.is_some()
                    && right_cut.is_some()
                    && left_cut.unwrap().1 != right_cut.unwrap().1
                {
                    cuts.push(right_cut.unwrap());
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
    fn test_contracted_tree_transformation() {
        let tree = sample_tree();
        let costs = vec![1.0, 3.0, 6.0];
        let contracted_tree = tree.contract_tree(costs);
        println!("{:?}", contracted_tree);
        assert_eq!(contracted_tree.probabilities(0), vec![0.0, 1.0, 0.0]);
        assert_eq!(contracted_tree.probabilities(7), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_contracted_tree() {
        let pool = sample_cut_pool();
        let cut_0 = Cut(0, CutOrientation::Normal);
        let cut_1 = Cut(1, CutOrientation::Normal);
        let cut_2 = Cut(2, CutOrientation::Normal);
        let cut_0_inv = Cut(0, CutOrientation::Inverted);
        let cut_1_inv = Cut(1, CutOrientation::Inverted);
        let cut_2_inv = Cut(2, CutOrientation::Inverted);
        let costs = vec![1.0, 3.0, 6.0];
        let mut tree =
            ContractedTanglesTree::new(costs, vec![cut_0, cut_1_inv, cut_2], CutPool::new(pool));
        assert_eq!(tree.nodes[0].cost_normalizer, 10.0);
        // reminder:
        // distinguishing cuts are always implicitly distinguishing the orientation
        // in the right subtree
        // let cut_0     = bits![1, 1, 1, 0, 0, 0, 0, 0, 0];
        // let cut_1_inv = bits![1, 1, 1, 0, 0, 0, 1, 1, 1];
        // let cut_2     = bits![0, 0, 0, 0, 0, 0, 1, 1, 1];
        tree.insert_children(0, vec![cut_0], vec![cut_0, cut_1_inv, cut_2]);
        // right_probability: (1.0 + 3.0 / 10.0) = 0.4
        assert_eq!(tree.probabilities(0), vec![0.6, 0.4]);
        assert_eq!(tree.probabilities(7), vec![0.09999999999999998, 0.9]);
        tree.insert_children(1, vec![], vec![]);
        // 0.0, 0.6, 0.4
        assert_eq!(tree.probabilities(0), vec![0.0, 0.6, 0.4]);
        assert_eq!(tree.probabilities(7), vec![0.09999999999999998, 0., 0.9]);
        tree.insert_children(2, vec![], vec![]);
        assert_eq!(
            tree.probabilities(0),
            vec![0.0, 0.6, 0.24, 0.16000000000000003]
        );
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
        assert_eq!(cuts, vec![Cut(0, CutOrientation::Inverted)]);
        let cuts = tree.distinguishing_cuts(1);
        assert_eq!(cuts, vec![Cut(1, CutOrientation::Inverted)]);
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
        cuts.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            cuts,
            vec![
                Cut(0, CutOrientation::Inverted),
                Cut(2, CutOrientation::Inverted)
            ]
        );
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
