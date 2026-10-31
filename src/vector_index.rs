//! Generic exact vector search. Metadata selection supplies IDs; no topic/tool taxonomy.
use crate::ranking::{score_cmp_desc, top_k};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

fn normalized(v: Vec<f32>, dim: usize) -> Result<Vec<f32>, String> {
    if v.len() != dim || v.iter().any(|x| !x.is_finite()) {
        return Err("embedding dimension mismatch or non-finite values".into());
    }
    let norm = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Err("zero embedding is not searchable".into());
    }
    Ok(v.into_iter()
        .map(|x| (f64::from(x) / norm) as f32)
        .collect())
}

#[pyclass(frozen, module = "memory_core._native")]
pub struct PyVectorIndex {
    dim: usize,
    vectors: RwLock<HashMap<String, Vec<f32>>>,
}

#[pymethods]
impl PyVectorIndex {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dimension must be positive"));
        }
        Ok(Self {
            dim,
            vectors: RwLock::new(HashMap::new()),
        })
    }
    fn upsert(&self, py: Python<'_>, id: String, vector: Vec<f32>) -> PyResult<()> {
        let vector = normalized(vector, self.dim).map_err(PyValueError::new_err)?;
        py.detach(|| {
            self.vectors
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(id, vector)
        });
        Ok(())
    }
    fn remove(&self, py: Python<'_>, id: String) -> bool {
        py.detach(|| {
            self.vectors
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&id)
                .is_some()
        })
    }
    fn search(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        ids: Vec<String>,
        k: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        let query = normalized(query, self.dim).map_err(PyValueError::new_err)?;
        Ok(py.detach(|| {
            let vectors = self.vectors.read().unwrap_or_else(|e| e.into_inner());
            let mut scores: Vec<_> = ids
                .into_iter()
                .filter_map(|id| {
                    vectors.get(&id).map(|v| {
                        let score: f32 = v.iter().zip(&query).map(|(a, b)| a * b).sum();
                        (id, score)
                    })
                })
                .collect();
            top_k(&mut scores, k, |a, b| {
                score_cmp_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0))
            });
            scores
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_and_normalizes() {
        assert!(normalized(vec![0.0, 0.0], 2).is_err());
        assert!(normalized(vec![f32::NAN], 1).is_err());
        assert!(normalized(vec![1.0], 2).is_err());
        assert_eq!(normalized(vec![3.0, 4.0], 2).unwrap(), vec![0.6, 0.8]);
    }
}
