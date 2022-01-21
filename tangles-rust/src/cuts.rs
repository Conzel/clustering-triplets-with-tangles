use std::ops::Not;

use bitvec::vec::BitVec;

/// A cut consists of a Bitarray, representing a bipartition
/// (if BitArray[i] == 1, then the i-th node is in the left side of the bipartition)])
/// bool represents the orientation of the cut (whether all elements in the bit array should be inverted
/// to get the actual cut or not)
pub type CutValue = BitVec;

/// We describe cuts with a combination of their ID (which can be used to get the
/// actual value from the Cut Pool) and the orientation.
#[derive(Clone, Copy, PartialEq)]
pub struct Cut(pub CutId, pub CutOrientation);

impl std::fmt::Debug for Cut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.1 {
            CutOrientation::Normal => write!(f, "Cut({})", self.0),
            CutOrientation::Inverted => write!(f, "-Cut({})", self.0),
        }
    }
}

/// Index into the array of all cuts. Also denotes whether the cut is saved in a
/// left or right oriented fashion.
pub type CutId = u16;

/// Holds all the cuts underlying a Tangle Tree.
/// The Cut Indices refer to the positions of the cuts in the CutPool
#[derive(Debug, Clone, PartialEq)]
pub struct CutPool {
    normal: Vec<CutValue>,
    inverted: Vec<CutValue>,
}

impl CutPool {
    pub fn new(normal_pool: Vec<CutValue>) -> CutPool {
        let inverted = normal_pool.iter().map(|cut| cut.clone().not()).collect();
        CutPool {
            normal: normal_pool,
            inverted,
        }
    }

    pub fn cut_value(&self, cut: Cut) -> &CutValue {
        match cut.1 {
            CutOrientation::Normal => &self.normal[cut.0 as usize],
            CutOrientation::Inverted => &self.inverted[cut.0 as usize],
        }
    }

    pub fn len(&self) -> usize {
        self.normal.len()
    }
}

/// Holds all the cuts that are relevant for consistency
pub type Core = Vec<Cut>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CutOrientation {
    /// Also called Left
    Normal,
    /// Also called Right
    Inverted,
}

/// Checks if a cut is consistent with the given core.
/// CutPool is used to determine the cut values.
/// A cut C is consistent with a core, if for all A, B in the core, we have that
///     |A ^ B ^ C| >= agreement
pub fn is_consistent(cut: Cut, core: &Core, pool: &CutPool, agreement: u16) -> bool {
    let cut_a = pool.cut_value(cut);
    // Checking for consistency
    if core.len() == 0 {
        return cut_a.count_ones() >= (agreement as usize);
    }
    if core.len() == 1 {
        let cut_b = pool.cut_value(core[0]);
        return intersection(cut_a, cut_b).count_ones() >= (agreement as usize);
    }
    for i in 0..core.len() {
        let cut_b = pool.cut_value(core[i]);
        let intersection_a_b = intersection(cut_a, cut_b);
        for j in i..core.len() {
            let cut_c = pool.cut_value(core[j]);
            let intersection_a_b_c = intersection(&intersection_a_b, cut_c);
            // count_ones gives cardinality of bitvec
            if intersection_a_b_c.count_ones() < agreement.into() {
                return false;
            }
        }
    }
    return true;
}

pub fn intersection(a: &BitVec, b: &BitVec) -> BitVec {
    // TODO: Remove the clone here
    a.clone() & b
}

pub fn is_subset(a: &BitVec, b: &BitVec) -> bool {
    intersection(a, b).eq(a)
}
