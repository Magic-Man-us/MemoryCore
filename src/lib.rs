use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::{HashMap, HashSet};

mod smk_index;

use smk_index::{
    RustMemoryIndex,
    SmkQuery,
    StructuredMemoryKey,
    TopicBucket,
    MemoryKind,
    Level2Bits,
};

#[derive(Clone)]
struct IndexedTrace {
    trace_uid: String,
    user_id: String,
    summary: String,
    importance: f32,
    created_at: i64, // unix timestamp (seconds)
    access_count: u32,
    tags: Vec<String>,
    deleted: bool,
}

/// Query specification used internally.
struct QuerySpec {
    user_id: String,
    text: String,
    tags: Vec<String>,
    limit: usize,
}

#[derive(Clone)]
struct Candidate {
    trace_uid: String,
    score: f32,
    summary: String,
    tags: Vec<String>,
    created_at: i64,
}

/// Main in-memory engine.
///
/// This is what we’ll later extend to talk to Redis/MariaDB.
struct MemoryEngine {
    traces: Vec<IndexedTrace>,
    embeddings: Vec<Vec<f32>>, // same index as traces

    // trace_uid -> index in traces / embeddings
    trace_index: HashMap<String, usize>,

    // tag -> indices
    tag_index: HashMap<String, Vec<usize>>,

    // keyword -> indices
    keyword_index: HashMap<String, Vec<usize>>,

    // embedding dimension (enforced)
    embedding_dim: Option<usize>,
}

impl MemoryEngine {
    fn new() -> Self {
        Self {
            traces: Vec::new(),
            embeddings: Vec::new(),
            trace_index: HashMap::new(),
            tag_index: HashMap::new(),
            keyword_index: HashMap::new(),
            embedding_dim: None,
        }
    }

    /// Simple tokenizer: lowercase and split on non-alphanumeric.
    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_alphanumeric() {
                current.push(ch.to_ascii_lowercase());
            } else if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Add or update a trace in the engine.
    fn ingest_trace(&mut self, trace: IndexedTrace, embedding: &[f32]) -> Result<(), String> {
        if embedding.is_empty() {
            return Err("embedding cannot be empty".into());
        }

        match self.embedding_dim {
            None => {
                self.embedding_dim = Some(embedding.len());
            }
            Some(dim) if dim != embedding.len() => {
                return Err(format!(
                    "embedding dimension mismatch: expected {}, got {}",
                    dim,
                    embedding.len()
                ));
            }
            _ => {}
        }

        // replace if exists
        if let Some(&idx) = self.trace_index.get(&trace.trace_uid) {
            self.traces[idx] = trace;
            self.embeddings[idx] = embedding.to_vec();
            self.reindex_entry(idx);
            return Ok(());
        }

        // new entry
        let idx = self.traces.len();
        self.trace_index.insert(trace.trace_uid.clone(), idx);
        self.traces.push(trace);
        self.embeddings.push(embedding.to_vec());

        self.index_entry(idx);

        Ok(())
    }

    /// Index a single entry into tag/keyword indexes.
    fn index_entry(&mut self, idx: usize) {
        if let Some(trace) = self.traces.get(idx) {
            if trace.deleted {
                return;
            }

            for tag in &trace.tags {
                self.tag_index
                    .entry(tag.to_ascii_lowercase())
                    .or_default()
                    .push(idx);
            }

            for token in Self::tokenize(&trace.summary) {
                self.keyword_index
                    .entry(token)
                    .or_default()
                    .push(idx);
            }
        }
    }

    /// Rebuild indexes for a given entry (naive: re-add).
    fn reindex_entry(&mut self, idx: usize) {
        // crude but fine for prototype: we don't remove old references,
        // we just tolerate duplicates and filter deleted later.
        self.index_entry(idx);
    }

    fn mark_accessed(&mut self, trace_uid: &str) {
        if let Some(&idx) = self.trace_index.get(trace_uid) {
            if let Some(trace) = self.traces.get_mut(idx) {
                if !trace.deleted {
                    trace.access_count = trace.access_count.saturating_add(1);
                }
            }
        }
    }

    fn remove_trace(&mut self, trace_uid: &str) {
        if let Some(&idx) = self.trace_index.get(trace_uid) {
            if let Some(trace) = self.traces.get_mut(idx) {
                trace.deleted = true;
            }
            // we leave indexes dirty; search filters deleted entries
        }
    }

    /// Simple cosine similarity.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    /// Core search logic.
    ///
    /// This does:
    /// 1. Filter by user
    /// 2. Seed candidates from tags + keywords
    /// 3. If candidate set is too small, fallback to all traces for that user
    /// 4. Score with semantic similarity + importance
    /// 5. Return top-k
    fn search_candidates(&self, query: QuerySpec, query_embedding: &[f32]) -> Vec<Candidate> {
        let mut candidate_indices: HashSet<usize> = HashSet::new();

        // 1. tags
        for tag in query.tags.iter() {
            if let Some(indices) = self.tag_index.get(&tag.to_ascii_lowercase()) {
                candidate_indices.extend(indices);
            }
        }

        // 2. keywords
        let tokens = Self::tokenize(&query.text);
        for token in tokens {
            if let Some(indices) = self.keyword_index.get(&token) {
                candidate_indices.extend(indices);
            }
        }

        // 3. If too few candidates, fall back to all traces for that user
        if candidate_indices.len() < query.limit {
            for (idx, trace) in self.traces.iter().enumerate() {
                if trace.deleted {
                    continue;
                }
                if trace.user_id == query.user_id {
                    candidate_indices.insert(idx);
                }
            }
        }

        // 4. Score candidates
        let mut scored: Vec<Candidate> = Vec::new();

        for idx in candidate_indices {
            if idx >= self.traces.len() || idx >= self.embeddings.len() {
                continue;
            }

            let trace = &self.traces[idx];
            if trace.deleted {
                continue;
            }
            if trace.user_id != query.user_id {
                continue;
            }

            let emb = &self.embeddings[idx];
            let semantic = Self::cosine_similarity(query_embedding, emb);
            let importance = trace.importance;

            // simple score function; you can tune weights
            let score = 0.7 * semantic + 0.3 * importance;

            scored.push(Candidate {
                trace_uid: trace.trace_uid.clone(),
                score,
                summary: trace.summary.clone(),
                tags: trace.tags.clone(),
                created_at: trace.created_at,
            });
        }

        // 5. Sort and take top-k
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(query.limit);

        scored
    }
}

#[pyclass]
pub struct PyMemoryCandidate {
    trace_uid: String,
    score: f32,
    summary: String,
    tags: Vec<String>,
    created_at: i64,
}

impl From<Candidate> for PyMemoryCandidate {
    fn from(c: Candidate) -> Self {
        PyMemoryCandidate {
            trace_uid: c.trace_uid,
            score: c.score,
            summary: c.summary,
            tags: c.tags,
            created_at: c.created_at,
        }
    }
}

#[pymethods]
impl PyMemoryCandidate {
    #[getter]
    fn trace_uid(&self) -> &str {
        &self.trace_uid
    }

    #[getter]
    fn score(&self) -> f32 {
        self.score
    }

    #[getter]
    fn summary(&self) -> &str {
        &self.summary
    }

    #[getter]
    fn tags(&self) -> Vec<String> {
        self.tags.clone()
    }

    #[getter]
    fn created_at(&self) -> i64 {
        self.created_at
    }
}

#[pyclass]
pub struct PyMemoryEngine {
    inner: MemoryEngine,
}

#[pymethods]
impl PyMemoryEngine {
    /// Create a new engine.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: MemoryEngine::new(),
        }
    }

    /// Ingest or update a trace.
    ///
    /// Parameters:
    /// - trace_uid: stable ID (UUID/ULID)
    /// - user_id: which user this belongs to
    /// - summary: short summary text (what LLM will mostly see)
    /// - importance: float 0..1
    /// - created_at: unix timestamp (seconds)
    /// - access_count: usage count
    /// - tags: list of tags
    /// - embedding: vector representation (same dim for all traces)
    pub fn ingest_trace(
        &mut self,
        trace_uid: String,
        user_id: String,
        summary: String,
        importance: f32,
        created_at: i64,
        access_count: u32,
        tags: Vec<String>,
        embedding: Vec<f32>,
    ) -> PyResult<()> {
        let trace = IndexedTrace {
            trace_uid,
            user_id,
            summary,
            importance,
            created_at,
            access_count,
            tags,
            deleted: false,
        };

        self.inner
            .ingest_trace(trace, &embedding)
            .map_err(|msg| pyo3::exceptions::PyValueError::new_err(msg))
    }

    /// Mark a trace as accessed (increments access_count internally).
    pub fn mark_accessed(&mut self, trace_uid: String) {
        self.inner.mark_accessed(&trace_uid);
    }

    /// Remove a trace (logically).
    pub fn remove_trace(&mut self, trace_uid: String) {
        self.inner.remove_trace(&trace_uid);
    }

    /// Search candidates for a given query.
    ///
    /// Returns a list of PyMemoryCandidate objects.
    pub fn search_candidates(
        &self,
        user_id: String,
        text: String,
        tags: Vec<String>,
        limit: usize,
        query_embedding: Vec<f32>,
    ) -> Vec<PyMemoryCandidate> {
        let q = QuerySpec {
            user_id,
            text,
            tags,
            limit,
        };

        self.inner
            .search_candidates(q, &query_embedding)
            .into_iter()
            .map(PyMemoryCandidate::from)
            .collect()
    }
}
#[pyclass]
pub struct PySmkQuery {
    inner: SmkQuery,
}

#[pymethods]
impl PySmkQuery {
    #[new]
    #[pyo3(signature = (
        topic = None,
        required_tools_mask = 0,
        allowed_kinds = None,
        min_generality = None,
        min_importance = None
    ))]
    pub fn new(
        topic: Option<u8>,
        required_tools_mask: u16,
        allowed_kinds: Option<Vec<u8>>,
        min_generality: Option<u8>,
        min_importance: Option<u8>,
    ) -> Self {
        let topic_enum = topic.map(|t| match t {
            1 => TopicBucket::RustPythonToolchain,
            2 => TopicBucket::MemoryArchitecture,
            3 => TopicBucket::AwsIam,
            4 => TopicBucket::DbSchema,
            _ => TopicBucket::RustPythonToolchain,
        });

        let kinds_enum = allowed_kinds.map(|ks| {
            ks.into_iter()
                .map(|k| match k {
                    0 => MemoryKind::Insight,
                    1 => MemoryKind::Pattern,
                    2 => MemoryKind::AntiPattern,
                    3 => MemoryKind::Principle,
                    _ => MemoryKind::Workflow,
                })
                .collect()
        });

        let min_gen = min_generality.map(|g| match g {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        });

        let min_imp = min_importance.map(|i| match i {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        });

        PySmkQuery {
            inner: SmkQuery {
                topic: topic_enum,
                required_tools_mask,
                allowed_kinds: kinds_enum,
                min_generality: min_gen,
                min_importance: min_imp,
            },
        }
    }
}

#[pyclass]
pub struct PyAssistantMemoryIndex {
    inner: RustMemoryIndex,
}

#[pymethods]
impl PyAssistantMemoryIndex {
    #[new]
    pub fn new(dim: usize) -> Self {
        PyAssistantMemoryIndex {
            inner: RustMemoryIndex::new(dim),
        }
    }

    /// Add a memory into the assistant index.
    ///
    /// `id` is a 64-bit internal ID (you can derive from trace_uid in Python).
    pub fn add(
        &mut self,
        id: u64,
        topic: u8,
        kind: u8,
        tool_mask: u16,
        difficulty: u8,
        generality: u8,
        importance: u8,
        embedding: Vec<f32>,
    ) {
        let topic_enum = match topic {
            1 => TopicBucket::RustPythonToolchain,
            2 => TopicBucket::MemoryArchitecture,
            3 => TopicBucket::AwsIam,
            4 => TopicBucket::DbSchema,
            _ => TopicBucket::RustPythonToolchain,
        };

        let kind_enum = match kind {
            0 => MemoryKind::Insight,
            1 => MemoryKind::Pattern,
            2 => MemoryKind::AntiPattern,
            3 => MemoryKind::Principle,
            _ => MemoryKind::Workflow,
        };

        let diff_enum = match difficulty {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        };

        let gen_enum = match generality {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        };

        let imp_enum = match importance {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        };

        let smk = StructuredMemoryKey::new(
            topic_enum,
            kind_enum,
            tool_mask,
            diff_enum,
            gen_enum,
            imp_enum,
        );

        let mem = smk_index::MemoryTrace { id, embedding, smk };
        self.inner.add(mem);
    }

    /// Query with a vector + SMK query.
    ///
    /// Returns a list of triples (id, score, smk_raw).
    pub fn query_top_k_filtered(
        &self,
        query: Vec<f32>,
        k: usize,
        smk_query: &PySmkQuery,
    ) -> Vec<(u64, f32, u64)> {
        self.inner
            .query_top_k_filtered(&query, k, &smk_query.inner)
            .into_iter()
            .map(|(id, score, smk)| (id, score, smk.raw()))
            .collect()
    }
}


#[pymodule]
fn memory_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // conversation/user memory
    m.add_class::<PyMemoryEngine>()?;
    m.add_class::<PyMemoryCandidate>()?;

    // assistant “brain” index
    m.add_class::<PyAssistantMemoryIndex>()?;
    m.add_class::<PySmkQuery>()?;

    Ok(())
}
