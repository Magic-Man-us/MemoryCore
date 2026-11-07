//! `memory_core._native`: exact vector ranking behind MemoryCore.
//!
//! [`PyVectorIndex`] ranks caller-selected IDs by cosine similarity. The index lives
//! behind an `RwLock`, and every lock acquisition happens with the GIL released
//! (`Python::detach`), so Python threads can search in parallel and a writer waiting
//! for the lock never stalls the interpreter.

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod ranking;
mod vector_index;
use vector_index::PyVectorIndex;

#[pymodule]
fn _native(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorIndex>()?;
    Ok(())
}
