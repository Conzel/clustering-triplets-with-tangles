use bitvec::bits;
use bitvec::prelude::Lsb0;

use crate::tangles::tangle_search_tree;

pub mod cuts;
pub mod tangles;

// TODO: Embed into a python structure

fn main() {
    // Assume 6 points that are clustered, such that
    // points 1-3 are in a cluster, 4-6 in another one,
    // points 7-9 in a third one.
    // Expected tangles:
    // c1,  -c2, -c3 (Cluster 1)
    // -c1,  c2, -c3 (Cluster 2)
    // -c1, -c2,  c3 (Cluster 3)
    let cut_1 = bits![1, 1, 1, 0, 0, 0, 0, 0, 0];
    let cut_2 = bits![0, 0, 0, 1, 1, 1, 0, 0, 0];
    let cut_3 = bits![0, 0, 0, 0, 0, 0, 1, 1, 1];
    let cuts = vec![cut_1.into(), cut_2.into(), cut_3.into()];
    let tree = tangle_search_tree(cuts, 3);
    tree.pretty_print();
    println!("{:?}", tree);
}
