//! `memory_core._native`: the Rust query engines behind MemoryCore.
//!
//! - [`PyMemoryEngine`]: per-user conversation memory. Tag/keyword candidate seeding,
//!   then a weighted score over cosine similarity, importance, recency, and access
//!   frequency.
//! - [`PyVectorIndex`]: generic exact vector ranking over caller-selected IDs.
//!
//! Concurrency: every engine lives behind an `RwLock`, and every lock acquisition
//! happens with the GIL released (`Python::detach`). Python threads (the stores run
//! under `asyncio.to_thread`) can therefore search in parallel, and a writer waiting
//! for the lock never stalls the interpreter.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::{HashMap, HashSet};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{SystemTime, UNIX_EPOCH};

mod ranking;
mod vector_index;
use ranking::{score_cmp_desc, top_k};
use vector_index::PyVectorIndex;

const SECONDS_PER_DAY: f64 = 86_400.0;

// --- Scoring ------------------------------------------------------------------

/// Weights for the candidate score:
///
/// `semantic * cosine + importance * importance + recency * 0.5^(age / half_life)
///  + access * a / (1 + a)` where `a = ln(1 + access_count)`.
///
/// Every term is in `[0, 1]` (cosine aside, which is in `[-1, 1]`), so the weights
/// read as relative influence. The defaults (0.7 / 0.3 / 0 / 0) are the original
/// semantic + importance score.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoreWeights {
    semantic: f32,
    importance: f32,
    recency: f32,
    access: f32,
    recency_half_life_secs: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            semantic: 0.7,
            importance: 0.3,
            recency: 0.0,
            access: 0.0,
            recency_half_life_secs: 30.0 * SECONDS_PER_DAY,
        }
    }
}

impl ScoreWeights {
    fn new(
        semantic: f32,
        importance: f32,
        recency: f32,
        access: f32,
        recency_half_life_days: f64,
    ) -> Result<Self, String> {
        for (name, w) in [
            ("semantic_weight", semantic),
            ("importance_weight", importance),
            ("recency_weight", recency),
            ("access_weight", access),
        ] {
            if !w.is_finite() || w < 0.0 {
                return Err(format!("{name} must be finite and >= 0, got {w}"));
            }
        }
        if !recency_half_life_days.is_finite() || recency_half_life_days <= 0.0 {
            return Err(format!(
                "recency_half_life_days must be finite and > 0, got {recency_half_life_days}"
            ));
        }
        Ok(Self {
            semantic,
            importance,
            recency,
            access,
            recency_half_life_secs: recency_half_life_days * SECONDS_PER_DAY,
        })
    }

    fn score(&self, semantic: f32, trace: &IndexedTrace, now: i64) -> f32 {
        let mut score = self.semantic * semantic + self.importance * trace.importance;
        if self.recency > 0.0 {
            let age = (now - trace.created_at).max(0) as f64;
            let decay = 0.5f64.powf(age / self.recency_half_life_secs) as f32;
            score += self.recency * decay;
        }
        if self.access > 0.0 {
            let a = (trace.access_count as f32).ln_1p();
            score += self.access * (a / (1.0 + a));
        }
        score
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// --- Conversation memory engine ----------------------------------------------

#[derive(Debug, Clone)]
struct IndexedTrace {
    trace_uid: String,
    user_id: String,
    summary: String,
    importance: f32,
    created_at: i64, // unix timestamp (seconds)
    access_count: u32,
    tags: Vec<String>,
}

/// Query specification used internally.
struct QuerySpec {
    user_id: String,
    text: String,
    tags: Vec<String>,
    limit: usize,
}

#[derive(Debug, Clone)]
struct Candidate {
    trace_uid: String,
    score: f32,
    summary: String,
    tags: Vec<String>,
    created_at: i64,
    importance: f32,
    access_count: u32,
}

type PostingIndex = HashMap<String, HashSet<usize>>;

fn post(index: &mut PostingIndex, key: String, idx: usize) {
    index.entry(key).or_default().insert(idx);
}

fn unpost(index: &mut PostingIndex, key: &str, idx: usize) {
    if let Some(set) = index.get_mut(key) {
        set.remove(&idx);
        if set.is_empty() {
            index.remove(key);
        }
    }
}

/// In-memory engine. Storage is dense (`traces[i]` pairs with `embeddings[i]`);
/// removal uses `swap_remove` and re-posts the moved entry, so there are no
/// tombstones and no stale posting-list entries.
#[derive(Default)]
struct MemoryEngine {
    traces: Vec<IndexedTrace>,
    embeddings: Vec<Vec<f32>>,
    trace_index: HashMap<String, usize>,
    user_index: PostingIndex,
    tag_index: PostingIndex,
    keyword_index: PostingIndex,
    embedding_dim: Option<usize>,
    weights: ScoreWeights,
}

impl MemoryEngine {
    fn with_weights(weights: ScoreWeights) -> Self {
        Self {
            weights,
            ..Self::default()
        }
    }

    /// Lowercase and split on non-alphanumeric characters.
    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() {
                current.extend(ch.to_lowercase());
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    fn normalize_tag(tag: &str) -> String {
        tag.to_lowercase()
    }

    /// Add `idx` to the user, tag, and keyword posting lists.
    fn link(&mut self, idx: usize) {
        let trace = &self.traces[idx];
        post(&mut self.user_index, trace.user_id.clone(), idx);
        for tag in &trace.tags {
            post(&mut self.tag_index, Self::normalize_tag(tag), idx);
        }
        for token in Self::tokenize(&trace.summary) {
            post(&mut self.keyword_index, token, idx);
        }
    }

    /// Remove `idx` from every posting list it appears in.
    fn unlink(&mut self, idx: usize) {
        let trace = &self.traces[idx];
        unpost(&mut self.user_index, &trace.user_id, idx);
        for tag in &trace.tags {
            unpost(&mut self.tag_index, &Self::normalize_tag(tag), idx);
        }
        for token in Self::tokenize(&trace.summary) {
            unpost(&mut self.keyword_index, &token, idx);
        }
    }

    /// Add or replace a trace. Replacement re-posts it, so an edited summary stops
    /// matching its old keywords.
    fn ingest_trace(&mut self, trace: IndexedTrace, embedding: Vec<f32>) -> Result<(), String> {
        if embedding.is_empty() {
            return Err("embedding cannot be empty".into());
        }
        if let Some(dim) = self.embedding_dim {
            if dim != embedding.len() {
                return Err(format!(
                    "embedding dimension mismatch: expected {}, got {}",
                    dim,
                    embedding.len()
                ));
            }
        }
        if !(0.0..=1.0).contains(&trace.importance) {
            return Err(format!(
                "importance must be in [0, 1], got {}",
                trace.importance
            ));
        }
        self.embedding_dim.get_or_insert(embedding.len());

        if let Some(&idx) = self.trace_index.get(&trace.trace_uid) {
            self.unlink(idx);
            self.traces[idx] = trace;
            self.embeddings[idx] = embedding;
            self.link(idx);
            return Ok(());
        }

        let idx = self.traces.len();
        self.trace_index.insert(trace.trace_uid.clone(), idx);
        self.traces.push(trace);
        self.embeddings.push(embedding);
        self.link(idx);
        Ok(())
    }

    fn mark_accessed(&mut self, trace_uid: &str) -> bool {
        match self.trace_index.get(trace_uid) {
            Some(&idx) => {
                let trace = &mut self.traces[idx];
                trace.access_count = trace.access_count.saturating_add(1);
                true
            }
            None => false,
        }
    }

    /// Remove a trace and free its storage. Returns `true` when it existed.
    fn remove_trace(&mut self, trace_uid: &str) -> bool {
        let Some(idx) = self.trace_index.remove(trace_uid) else {
            return false;
        };
        let last = self.traces.len() - 1;
        self.unlink(idx);
        if idx != last {
            // The last entry is about to move into `idx`: re-post it at its new slot.
            self.unlink(last);
        }
        self.traces.swap_remove(idx);
        self.embeddings.swap_remove(idx);
        if idx != last {
            self.trace_index
                .insert(self.traces[idx].trace_uid.clone(), idx);
            self.link(idx);
        }
        true
    }

    fn len(&self) -> usize {
        self.traces.len()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

    /// Core search:
    /// 1. Search all of this user's vectors so lexical matches cannot hide semantic hits.
    /// 2. Extensible retrieval combines separate semantic and lexical result lists.
    /// 3. Score with the configured weights; return the top `limit`, ties broken by
    ///    newer first, then trace uid.
    fn search_candidates(
        &self,
        query: QuerySpec,
        query_embedding: &[f32],
        now: i64,
    ) -> Result<Vec<Candidate>, String> {
        if let Some(dim) = self.embedding_dim {
            if query_embedding.len() != dim {
                return Err(format!(
                    "query embedding dimension mismatch: expected {}, got {}",
                    dim,
                    query_embedding.len()
                ));
            }
        }
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        let Some(user_traces) = self.user_index.get(&query.user_id) else {
            return Ok(Vec::new());
        };

        // Semantic retrieval must not be suppressed by keyword matches.
        // Structured filtering belongs to the metadata index; these are ranking cues.
        let _lexical_context = (&query.text, &query.tags);
        let pool = user_traces.iter().copied();

        let mut scored: Vec<Candidate> = pool
            .map(|idx| {
                let trace = &self.traces[idx];
                let semantic = Self::cosine_similarity(query_embedding, &self.embeddings[idx]);
                Candidate {
                    trace_uid: trace.trace_uid.clone(),
                    score: self.weights.score(semantic, trace, now),
                    summary: trace.summary.clone(),
                    tags: trace.tags.clone(),
                    created_at: trace.created_at,
                    importance: trace.importance,
                    access_count: trace.access_count,
                }
            })
            .collect();

        // NaN-safe under a true total order (see score_cmp_desc): a non-finite
        // score sinks to the bottom rather than scattering through the results.
        top_k(&mut scored, query.limit, |a, b| {
            score_cmp_desc(a.score, b.score)
                .then_with(|| b.created_at.cmp(&a.created_at))
                .then_with(|| a.trace_uid.cmp(&b.trace_uid))
        });
        Ok(scored)
    }
}

fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    // A panic while holding the lock cannot leave these engines half-updated in a
    // way later reads could not tolerate, so recover from poisoning.
    lock.read().unwrap_or_else(|e| e.into_inner())
}

fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

// --- Python bindings: conversation memory -------------------------------------

#[pyclass(frozen, get_all, module = "memory_core._native")]
pub struct PyMemoryCandidate {
    trace_uid: String,
    score: f32,
    summary: String,
    tags: Vec<String>,
    created_at: i64,
    importance: f32,
    access_count: u32,
}

impl From<Candidate> for PyMemoryCandidate {
    fn from(c: Candidate) -> Self {
        PyMemoryCandidate {
            trace_uid: c.trace_uid,
            score: c.score,
            summary: c.summary,
            tags: c.tags,
            created_at: c.created_at,
            importance: c.importance,
            access_count: c.access_count,
        }
    }
}

#[pyclass(frozen, module = "memory_core._native")]
pub struct PyMemoryEngine {
    inner: RwLock<MemoryEngine>,
}

#[pymethods]
impl PyMemoryEngine {
    /// Create an engine. Weights must be finite and non-negative; the defaults are
    /// the original `0.7 * cosine + 0.3 * importance` score.
    #[new]
    #[pyo3(signature = (
        semantic_weight = 0.7,
        importance_weight = 0.3,
        recency_weight = 0.0,
        access_weight = 0.0,
        recency_half_life_days = 30.0
    ))]
    pub fn new(
        semantic_weight: f32,
        importance_weight: f32,
        recency_weight: f32,
        access_weight: f32,
        recency_half_life_days: f64,
    ) -> PyResult<Self> {
        let weights = ScoreWeights::new(
            semantic_weight,
            importance_weight,
            recency_weight,
            access_weight,
            recency_half_life_days,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: RwLock::new(MemoryEngine::with_weights(weights)),
        })
    }

    /// Ingest or replace a trace.
    ///
    /// Raises ValueError on an empty embedding, a dimension mismatch, or an
    /// importance outside [0, 1].
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_trace(
        &self,
        py: Python<'_>,
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
        };
        py.detach(|| write_lock(&self.inner).ingest_trace(trace, embedding))
            .map_err(PyValueError::new_err)
    }

    /// Increment a trace's access count. Returns False when the trace is unknown.
    pub fn mark_accessed(&self, py: Python<'_>, trace_uid: String) -> bool {
        py.detach(|| write_lock(&self.inner).mark_accessed(&trace_uid))
    }

    /// Remove a trace and free its memory. Returns False when the trace is unknown.
    pub fn remove_trace(&self, py: Python<'_>, trace_uid: String) -> bool {
        py.detach(|| write_lock(&self.inner).remove_trace(&trace_uid))
    }

    /// Search a user's traces. `now` (unix seconds) defaults to the system clock and
    /// only matters when the recency weight is non-zero.
    ///
    /// Raises ValueError on a query-embedding dimension mismatch.
    #[pyo3(signature = (user_id, text, tags, limit, query_embedding, now = None))]
    #[allow(clippy::too_many_arguments)]
    pub fn search_candidates(
        &self,
        py: Python<'_>,
        user_id: String,
        text: String,
        tags: Vec<String>,
        limit: usize,
        query_embedding: Vec<f32>,
        now: Option<i64>,
    ) -> PyResult<Vec<PyMemoryCandidate>> {
        let query = QuerySpec {
            user_id,
            text,
            tags,
            limit,
        };
        let now = now.unwrap_or_else(unix_now);
        py.detach(|| read_lock(&self.inner).search_candidates(query, &query_embedding, now))
            .map(|candidates| {
                candidates
                    .into_iter()
                    .map(PyMemoryCandidate::from)
                    .collect()
            })
            .map_err(PyValueError::new_err)
    }

    pub fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| read_lock(&self.inner).len())
    }

    /// The embedding dimension fixed by the first ingest, or None before it.
    /// Lets callers validate a vector before persisting anything.
    #[getter]
    pub fn embedding_dim(&self, py: Python<'_>) -> Option<usize> {
        py.detach(|| read_lock(&self.inner).embedding_dim)
    }
}

#[pymodule]
fn _native(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // conversation/user memory
    m.add_class::<PyMemoryEngine>()?;
    m.add_class::<PyMemoryCandidate>()?;

    m.add_class::<PyVectorIndex>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: i64 = 86_400;
    const NOW: i64 = 1_700_000_000;

    fn trace(
        uid: &str,
        user: &str,
        summary: &str,
        importance: f32,
        created_at: i64,
    ) -> IndexedTrace {
        IndexedTrace {
            trace_uid: uid.into(),
            user_id: user.into(),
            summary: summary.into(),
            importance,
            created_at,
            access_count: 0,
            tags: vec![],
        }
    }

    fn query(user: &str, text: &str, limit: usize) -> QuerySpec {
        QuerySpec {
            user_id: user.into(),
            text: text.into(),
            tags: vec![],
            limit,
        }
    }

    fn uids(c: &[Candidate]) -> Vec<&str> {
        c.iter().map(|c| c.trace_uid.as_str()).collect()
    }

    #[test]
    fn remove_frees_storage_and_keeps_moved_entry_searchable() {
        let mut e = MemoryEngine::default();
        e.ingest_trace(trace("a", "u", "alpha", 0.5, NOW), vec![1.0, 0.0])
            .unwrap();
        e.ingest_trace(trace("b", "u", "beta", 0.5, NOW), vec![0.0, 1.0])
            .unwrap();
        e.ingest_trace(trace("c", "u", "gamma", 0.5, NOW), vec![1.0, 1.0])
            .unwrap();

        assert!(e.remove_trace("a"));
        assert!(!e.remove_trace("a"));
        assert_eq!(e.len(), 2);
        assert!(!e.keyword_index.contains_key("alpha"), "no stale postings");

        // "c" moved into slot 0: its keyword must point at the new slot.
        let hits = e
            .search_candidates(query("u", "gamma", 1), &[1.0, 1.0], NOW)
            .unwrap();
        assert_eq!(uids(&hits), vec!["c"]);
        assert!(e.mark_accessed("c"));
        assert!(e.remove_trace("c"));
        assert_eq!(e.len(), 1);
        assert_eq!(e.keyword_index.get("beta").map(|s| s.len()), Some(1));
    }

    #[test]
    fn removing_the_last_user_trace_drops_the_user_posting() {
        let mut e = MemoryEngine::default();
        e.ingest_trace(trace("a", "u", "x", 0.5, NOW), vec![1.0])
            .unwrap();
        assert!(e.remove_trace("a"));
        assert!(e.user_index.is_empty());
        assert!(e
            .search_candidates(query("u", "x", 5), &[1.0], NOW)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn update_drops_old_keywords() {
        let mut e = MemoryEngine::default();
        e.ingest_trace(trace("a", "u", "likes tea", 0.5, NOW), vec![1.0])
            .unwrap();
        e.ingest_trace(trace("a", "u", "likes coffee", 0.5, NOW), vec![1.0])
            .unwrap();
        assert_eq!(e.len(), 1);
        assert!(!e.keyword_index.contains_key("tea"));
        assert!(e.keyword_index.contains_key("coffee"));
    }

    #[test]
    fn other_users_keyword_matches_do_not_suppress_fallback() {
        let mut e = MemoryEngine::default();
        for i in 0..5 {
            e.ingest_trace(
                trace(&format!("b{i}"), "bob", "python tip", 0.5, NOW),
                vec![1.0],
            )
            .unwrap();
        }
        e.ingest_trace(trace("a1", "alice", "snake case", 0.9, NOW), vec![1.0])
            .unwrap();
        e.ingest_trace(trace("a2", "alice", "short meetings", 0.8, NOW), vec![1.0])
            .unwrap();
        let hits = e
            .search_candidates(query("alice", "python", 3), &[1.0], NOW)
            .unwrap();
        assert_eq!(uids(&hits), vec!["a1", "a2"]);
    }

    #[test]
    fn invalid_inputs_are_errors() {
        let mut e = MemoryEngine::default();
        assert!(e
            .ingest_trace(trace("a", "u", "s", 0.5, NOW), vec![])
            .is_err());
        assert!(e
            .ingest_trace(trace("a", "u", "s", 7.3, NOW), vec![1.0])
            .is_err());
        assert!(e
            .ingest_trace(trace("a", "u", "s", f32::NAN, NOW), vec![1.0])
            .is_err());
        e.ingest_trace(trace("a", "u", "s", 0.5, NOW), vec![1.0, 0.0])
            .unwrap();
        assert!(e
            .ingest_trace(trace("b", "u", "s", 0.5, NOW), vec![1.0])
            .is_err());
        assert!(e
            .search_candidates(query("u", "s", 5), &[1.0], NOW)
            .is_err());
        assert!(e.search_candidates(query("u", "s", 5), &[], NOW).is_err());
    }

    #[test]
    fn weights_are_validated() {
        assert!(ScoreWeights::new(-0.1, 0.3, 0.0, 0.0, 30.0).is_err());
        assert!(ScoreWeights::new(0.7, f32::NAN, 0.0, 0.0, 30.0).is_err());
        assert!(ScoreWeights::new(0.7, 0.3, 0.0, 0.0, 0.0).is_err());
        assert!(ScoreWeights::new(0.7, 0.3, 0.1, 0.1, 7.0).is_ok());
    }

    #[test]
    fn default_weights_ignore_recency_and_access() {
        let mut e = MemoryEngine::default();
        let mut old = trace("old", "u", "s", 0.5, NOW - 365 * DAY);
        old.access_count = 50;
        e.ingest_trace(old, vec![1.0]).unwrap();
        e.ingest_trace(trace("new", "u", "s", 0.5, NOW), vec![1.0])
            .unwrap();
        let hits = e.search_candidates(query("u", "", 5), &[1.0], NOW).unwrap();
        assert!((hits[0].score - hits[1].score).abs() < 1e-6);
        // equal scores: newer first
        assert_eq!(uids(&hits), vec!["new", "old"]);
    }

    #[test]
    fn recency_weight_prefers_newer_traces_with_half_life_decay() {
        let w = ScoreWeights::new(0.0, 0.0, 1.0, 0.0, 10.0).unwrap();
        let mut e = MemoryEngine::with_weights(w);
        e.ingest_trace(trace("old", "u", "s", 0.5, NOW - 10 * DAY), vec![1.0])
            .unwrap();
        e.ingest_trace(trace("new", "u", "s", 0.5, NOW), vec![1.0])
            .unwrap();
        e.ingest_trace(trace("future", "u", "s", 0.5, NOW + DAY), vec![1.0])
            .unwrap();
        let hits = e.search_candidates(query("u", "", 5), &[1.0], NOW).unwrap();
        let score = |uid: &str| hits.iter().find(|c| c.trace_uid == uid).unwrap().score;
        assert!((score("new") - 1.0).abs() < 1e-6);
        assert!(
            (score("old") - 0.5).abs() < 1e-6,
            "one half-life halves the term"
        );
        assert!(
            (score("future") - 1.0).abs() < 1e-6,
            "clock skew clamps to age 0"
        );
    }

    #[test]
    fn access_weight_rewards_frequently_recalled_traces() {
        let w = ScoreWeights::new(0.5, 0.0, 0.0, 0.5, 30.0).unwrap();
        let mut e = MemoryEngine::with_weights(w);
        e.ingest_trace(trace("cold", "u", "s", 0.5, NOW), vec![1.0])
            .unwrap();
        e.ingest_trace(trace("hot", "u", "s", 0.5, NOW), vec![1.0])
            .unwrap();
        for _ in 0..10 {
            e.mark_accessed("hot");
        }
        let hits = e.search_candidates(query("u", "", 5), &[1.0], NOW).unwrap();
        assert_eq!(uids(&hits), vec!["hot", "cold"]);
        assert_eq!(hits[0].access_count, 10);
        assert!(
            hits[0].score < 1.0,
            "access term is bounded below its weight"
        );
    }

    #[test]
    fn tags_and_tokens_are_case_insensitive() {
        let mut e = MemoryEngine::default();
        let mut t = trace("a", "u", "Ärger mit RUST", 0.5, NOW);
        t.tags = vec!["Lang".into()];
        e.ingest_trace(t, vec![1.0]).unwrap();
        assert!(e.keyword_index.contains_key("ärger"));
        assert!(e.keyword_index.contains_key("rust"));
        assert!(e.tag_index.contains_key("lang"));
    }
}
