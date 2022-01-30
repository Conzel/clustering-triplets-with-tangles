use bitvec::prelude::BitVec;
use numpy::ndarray::{s, ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
mod cuts;
mod tangles;
use crate::tangles::{tangle_search_tree, ContractedTanglesTree};

#[pyclass(name = "TanglesTree", unsendable)]
struct ContractedTanglesTreePy {
    internal: ContractedTanglesTree,
}

#[pymethods]
impl ContractedTanglesTreePy {
    #[new]
    fn new(
        cuts: &PyArray2<bool>,
        costs: &PyArray1<f64>,
        agreement: u16,
        prune: u16,
    ) -> PyResult<ContractedTanglesTreePy> {
        if cuts.shape()[0] != costs.shape()[0] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                "Number of cuts must be equal to length of cost array. Got cuts with shape {:?} and cost array with shape {:?}",
                cuts.shape(), costs.shape())
            ));
        }
        let num_cuts = cuts.shape()[0];
        let mut cuts_bitvec: Vec<BitVec> = Vec::new();
        unsafe {
            let cut_array = cuts.as_array();
            for i in 0..num_cuts {
                let cut_bitvec = BitVec::from_iter(cut_array.slice(s![i, ..]));
                cuts_bitvec.push(cut_bitvec);
            }
        }
        let mut normal_tangle_tree = tangle_search_tree(cuts_bitvec, agreement);
        normal_tangle_tree.prune(prune);
        let mut cost_vec = Vec::new();
        unsafe {
            let cost_array = costs.as_array();
            for val in cost_array {
                cost_vec.push(*val);
            }
        }
        let contracted = normal_tangle_tree.contract_tree(cost_vec);
        Ok(ContractedTanglesTreePy {
            internal: contracted,
        })
    }

    fn probabilities(&self, v: u16) -> PyResult<Vec<f64>> {
        Ok(self.internal.probabilities(v))
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn tangles_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // immutable example
    m.add_class::<ContractedTanglesTreePy>()?;
    Ok(())
}
