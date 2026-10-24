//! Structured Memory Key (SMK) assistant index.
//!
//! A 64-bit packed key carries structured metadata about an assistant-level memory;
//! queries prune candidates with cheap bitfield checks before paying for cosine
//! similarity.
//!
//! Bit layout:
//!
//! ```text
//! bits  0-7:   topic       (u8, 1..=255; 0 is reserved)
//! bits  8-10:  kind        (MemoryKind)
//! bits 11-26:  tool_mask   (16-bit flags)
//! bits 27-28:  difficulty  (Level2Bits)
//! bits 29-30:  generality  (Level2Bits)
//! bits 31-32:  importance  (Level2Bits)
//! ```
//!
//! Topic and tool *meanings* live only in Python (`TopicBucket`, `ToolFlag`); Rust
//! treats them as opaque validated integers so the taxonomy can grow without a
//! cross-language edit.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

/// Descending comparator for similarity scores that defines a true total order:
/// bare `partial_cmp(...).unwrap_or(Equal)` treats every NaN as tied with every
/// finite score, which breaks `sort_by`'s ordering contract and can leave NaN
/// hits scattered through (or even ahead of) valid results. NaN is explicitly
/// the worst score and always sinks to the end, regardless of its sign bit;
/// finite scores use `total_cmp` for a well-defined order among themselves.
pub(crate) fn score_cmp_desc(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater, // a is NaN: a sorts after (worse than) b
        (false, true) => Ordering::Less,    // b is NaN: b sorts after (worse than) a
        (false, false) => b.total_cmp(&a),  // descending among finite scores
    }
}

/// Keep the best `k` items under `cmp` (which must be a total order), sorted.
///
/// Partial selection first, so a large index with a small `k` does not pay for a
/// full sort.
pub(crate) fn top_k<T, F>(items: &mut Vec<T>, k: usize, mut cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    if k == 0 {
        items.clear();
        return;
    }
    if items.len() > k {
        items.select_nth_unstable_by(k - 1, &mut cmp);
        items.truncate(k);
    }
    items.sort_by(cmp);
}

/// Validate a topic discriminant. Any non-zero u8 is a valid topic.
pub fn validate_topic(topic: u8) -> Result<u8, String> {
    if topic == 0 {
        Err("invalid topic discriminant: 0 is reserved (expected 1..=255)".into())
    } else {
        Ok(topic)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryKind {
    Insight = 0,
    Pattern = 1,
    AntiPattern = 2,
    Principle = 3,
    Workflow = 4,
}

impl TryFrom<u8> for MemoryKind {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(MemoryKind::Insight),
            1 => Ok(MemoryKind::Pattern),
            2 => Ok(MemoryKind::AntiPattern),
            3 => Ok(MemoryKind::Principle),
            4 => Ok(MemoryKind::Workflow),
            other => Err(format!("invalid kind discriminant: {other} (expected 0..=4)")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level2Bits {
    Low = 0,
    Medium = 1,
    High = 2,
    Extreme = 3,
}

impl Level2Bits {
    /// Total decode from two bits (every 2-bit value is a valid level).
    fn from_bits(bits: u64) -> Self {
        match bits & 0x3 {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        }
    }
}

impl TryFrom<u8> for Level2Bits {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0..=3 => Ok(Level2Bits::from_bits(u64::from(value))),
            other => Err(format!("invalid level discriminant: {other} (expected 0..=3)")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredMemoryKey(u64);

impl StructuredMemoryKey {
    /// Pack a key. Inputs are already validated by their types, except `topic`,
    /// which callers validate with [`validate_topic`].
    pub fn new(
        topic: u8,
        kind: MemoryKind,
        tool_mask: u16,
        difficulty: Level2Bits,
        generality: Level2Bits,
        importance: Level2Bits,
    ) -> Self {
        let mut v: u64 = u64::from(topic); // bits 0..=7
        v |= (kind as u64 & 0x7) << 8; // bits 8..=10
        v |= u64::from(tool_mask) << 11; // bits 11..=26
        v |= (difficulty as u64 & 0x3) << 27; // bits 27..=28
        v |= (generality as u64 & 0x3) << 29; // bits 29..=30
        v |= (importance as u64 & 0x3) << 31; // bits 31..=32
        StructuredMemoryKey(v)
    }

    pub fn raw(&self) -> u64 {
        self.0
    }

    pub fn topic(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    /// The packed kind. Keys are only ever built from a valid `MemoryKind`, so the
    /// fallback arm is unreachable in practice.
    pub fn kind(&self) -> MemoryKind {
        MemoryKind::try_from(((self.0 >> 8) & 0x7) as u8).unwrap_or(MemoryKind::Insight)
    }

    pub fn tool_mask(&self) -> u16 {
        ((self.0 >> 11) & 0xFFFF) as u16
    }

    pub fn difficulty(&self) -> Level2Bits {
        Level2Bits::from_bits(self.0 >> 27)
    }

    pub fn generality(&self) -> Level2Bits {
        Level2Bits::from_bits(self.0 >> 29)
    }

    pub fn importance(&self) -> Level2Bits {
        Level2Bits::from_bits(self.0 >> 31)
    }
}

impl fmt::Display for StructuredMemoryKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind_str = match self.kind() {
            MemoryKind::Insight => "INS",
            MemoryKind::Pattern => "PAT",
            MemoryKind::AntiPattern => "ANTI",
            MemoryKind::Principle => "PRIN",
            MemoryKind::Workflow => "WF",
        };
        let lvl = |l: Level2Bits| match l {
            Level2Bits::Low => "LO",
            Level2Bits::Medium => "MD",
            Level2Bits::High => "HI",
            Level2Bits::Extreme => "XHI",
        };
        write!(
            f,
            "SMK:V1 T-{} TO-{:#06x} K-{} G-{} D-{} I-{}",
            self.topic(),
            self.tool_mask(),
            kind_str,
            lvl(self.generality()),
            lvl(self.difficulty()),
            lvl(self.importance()),
        )
    }
}

/// Query-time filter that uses the SMK to prune candidates before cosine.
#[derive(Debug, Clone, Default)]
pub struct SmkQuery {
    pub topic: Option<u8>,
    pub required_tools_mask: u16,
    pub allowed_kinds: Option<Vec<MemoryKind>>,
    pub min_generality: Option<Level2Bits>,
    pub min_importance: Option<Level2Bits>,
}

impl SmkQuery {
    pub fn matches(&self, smk: &StructuredMemoryKey) -> bool {
        if let Some(t) = self.topic {
            if smk.topic() != t {
                return false;
            }
        }
        if self.required_tools_mask != 0
            && (smk.tool_mask() & self.required_tools_mask) != self.required_tools_mask
        {
            return false;
        }
        if let Some(ref kinds) = self.allowed_kinds {
            if !kinds.contains(&smk.kind()) {
                return false;
            }
        }
        if let Some(min_g) = self.min_generality {
            if smk.generality() < min_g {
                return false;
            }
        }
        if let Some(min_i) = self.min_importance {
            if smk.importance() < min_i {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct SmkEntry {
    pub id: u64,
    pub embedding: Vec<f32>,
    pub smk: StructuredMemoryKey,
}

/// Fixed-dimension index of SMK entries, keyed by a caller-supplied 64-bit id.
///
/// Ids are unique: adding an existing id replaces it in place (upsert), and
/// removal compacts storage with `swap_remove`, so removed entries free their RAM.
pub struct SmkIndex {
    dim: usize,
    entries: Vec<SmkEntry>,
    positions: HashMap<u64, usize>,
}

impl SmkIndex {
    pub fn new(dim: usize) -> Result<Self, String> {
        if dim == 0 {
            return Err("index dimension must be >= 1".into());
        }
        Ok(Self {
            dim,
            entries: Vec::new(),
            positions: HashMap::new(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Insert or replace an entry. Returns `true` when an existing id was replaced.
    pub fn upsert(&mut self, entry: SmkEntry) -> Result<bool, String> {
        if entry.embedding.len() != self.dim {
            return Err(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dim,
                entry.embedding.len()
            ));
        }
        if let Some(&pos) = self.positions.get(&entry.id) {
            self.entries[pos] = entry;
            return Ok(true);
        }
        self.positions.insert(entry.id, self.entries.len());
        self.entries.push(entry);
        Ok(false)
    }

    /// Remove an entry by id. Returns `true` when it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(pos) = self.positions.remove(&id) else {
            return false;
        };
        self.entries.swap_remove(pos);
        if let Some(moved) = self.entries.get(pos) {
            self.positions.insert(moved.id, pos);
        }
        true
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
        dot / ((norm_a.sqrt() * norm_b.sqrt()) + 1e-9)
    }

    /// Bitfield-filter, then rank by cosine. Ties break on id for determinism.
    pub fn query_top_k_filtered(
        &self,
        query: &[f32],
        k: usize,
        smk_query: &SmkQuery,
    ) -> Result<Vec<(u64, f32, StructuredMemoryKey)>, String> {
        if query.len() != self.dim {
            return Err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            ));
        }

        let mut scores: Vec<(u64, f32, StructuredMemoryKey)> = self
            .entries
            .iter()
            .filter(|m| smk_query.matches(&m.smk))
            .map(|m| (m.id, Self::cosine_similarity(&m.embedding, query), m.smk))
            .collect();

        top_k(&mut scores, k, |a, b| {
            score_cmp_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0))
        });
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(topic: u8, kind: MemoryKind, tools: u16, d: Level2Bits, g: Level2Bits, i: Level2Bits) -> StructuredMemoryKey {
        StructuredMemoryKey::new(topic, kind, tools, d, g, i)
    }

    fn entry(id: u64, embedding: Vec<f32>, smk: StructuredMemoryKey) -> SmkEntry {
        SmkEntry { id, embedding, smk }
    }

    fn default_key() -> StructuredMemoryKey {
        key(1, MemoryKind::Pattern, 0, Level2Bits::Low, Level2Bits::Low, Level2Bits::Low)
    }

    #[test]
    fn packing_matches_documented_layout() {
        let k = key(1, MemoryKind::Pattern, 0b11, Level2Bits::High, Level2Bits::High, Level2Bits::High);
        assert_eq!(k.raw(), 5_637_150_977);
        assert_eq!(k.topic(), 1);
        assert_eq!(k.kind(), MemoryKind::Pattern);
        assert_eq!(k.tool_mask(), 0b11);
        assert_eq!(k.difficulty(), Level2Bits::High);
        assert_eq!(k.generality(), Level2Bits::High);
        assert_eq!(k.importance(), Level2Bits::High);
    }

    #[test]
    fn every_field_round_trips_at_its_extremes() {
        let k = key(255, MemoryKind::Workflow, 0xFFFF, Level2Bits::Extreme, Level2Bits::Low, Level2Bits::Extreme);
        assert_eq!(k.topic(), 255);
        assert_eq!(k.kind(), MemoryKind::Workflow);
        assert_eq!(k.tool_mask(), 0xFFFF);
        assert_eq!(k.difficulty(), Level2Bits::Extreme);
        assert_eq!(k.generality(), Level2Bits::Low);
        assert_eq!(k.importance(), Level2Bits::Extreme);
        assert!(k.raw() < (1u64 << 33));
    }

    #[test]
    fn topic_zero_is_reserved_any_other_u8_is_valid() {
        assert!(validate_topic(0).is_err());
        assert_eq!(validate_topic(1), Ok(1));
        assert_eq!(validate_topic(255), Ok(255));
    }

    #[test]
    fn enum_discriminants_are_validated() {
        assert!(MemoryKind::try_from(5).is_err());
        assert!(Level2Bits::try_from(4).is_err());
        assert_eq!(Level2Bits::try_from(2), Ok(Level2Bits::High));
    }

    #[test]
    fn zero_dimension_is_rejected() {
        assert!(SmkIndex::new(0).is_err());
    }

    #[test]
    fn upsert_replaces_instead_of_duplicating() {
        let mut idx = SmkIndex::new(2).unwrap();
        assert_eq!(idx.upsert(entry(7, vec![1.0, 0.0], default_key())), Ok(false));
        assert_eq!(idx.upsert(entry(7, vec![0.0, 1.0], default_key())), Ok(true));
        assert_eq!(idx.len(), 1);
        let hits = idx.query_top_k_filtered(&[0.0, 1.0], 5, &SmkQuery::default()).unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].1 - 1.0).abs() < 1e-5, "the replacement embedding is the live one");
    }

    #[test]
    fn remove_compacts_and_keeps_the_rest_addressable() {
        let mut idx = SmkIndex::new(2).unwrap();
        for id in 1..=3 {
            idx.upsert(entry(id, vec![id as f32, 1.0], default_key())).unwrap();
        }
        assert!(idx.remove(1));
        assert!(!idx.remove(1));
        assert_eq!(idx.len(), 2);
        // id 3 was swapped into slot 0; it must still be replaceable and removable.
        assert_eq!(idx.upsert(entry(3, vec![0.0, 1.0], default_key())), Ok(true));
        assert!(idx.remove(3));
        let ids: Vec<u64> = idx
            .query_top_k_filtered(&[1.0, 1.0], 5, &SmkQuery::default())
            .unwrap()
            .into_iter()
            .map(|h| h.0)
            .collect();
        assert_eq!(ids, vec![2]);
    }

    #[test]
    fn filters_prune_before_scoring() {
        let mut idx = SmkIndex::new(2).unwrap();
        let e = vec![1.0, 0.0];
        idx.upsert(entry(1, e.clone(), key(1, MemoryKind::Pattern, 0b011, Level2Bits::Low, Level2Bits::High, Level2Bits::High))).unwrap();
        idx.upsert(entry(2, e.clone(), key(1, MemoryKind::Insight, 0b011, Level2Bits::Low, Level2Bits::High, Level2Bits::High))).unwrap();
        idx.upsert(entry(3, e.clone(), key(2, MemoryKind::Pattern, 0b011, Level2Bits::Low, Level2Bits::High, Level2Bits::High))).unwrap();
        idx.upsert(entry(4, e.clone(), key(1, MemoryKind::Pattern, 0b001, Level2Bits::Low, Level2Bits::High, Level2Bits::High))).unwrap();
        idx.upsert(entry(5, e.clone(), key(1, MemoryKind::Pattern, 0b011, Level2Bits::Low, Level2Bits::Low, Level2Bits::High))).unwrap();

        let q = SmkQuery {
            topic: Some(1),
            required_tools_mask: 0b011,
            allowed_kinds: Some(vec![MemoryKind::Pattern]),
            min_generality: Some(Level2Bits::Medium),
            min_importance: Some(Level2Bits::Medium),
        };
        let ids: Vec<u64> = idx.query_top_k_filtered(&e, 10, &q).unwrap().into_iter().map(|h| h.0).collect();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn ranking_is_deterministic_and_nan_sinks() {
        let mut idx = SmkIndex::new(2).unwrap();
        idx.upsert(entry(9, vec![f32::NAN, f32::NAN], default_key())).unwrap();
        idx.upsert(entry(3, vec![1.0, 0.0], default_key())).unwrap();
        idx.upsert(entry(1, vec![1.0, 0.0], default_key())).unwrap();
        idx.upsert(entry(2, vec![0.0, 1.0], default_key())).unwrap();
        let ids: Vec<u64> = idx
            .query_top_k_filtered(&[1.0, 0.0], 10, &SmkQuery::default())
            .unwrap()
            .into_iter()
            .map(|h| h.0)
            .collect();
        // equal scores (ids 1 and 3) break on id; NaN last
        assert_eq!(ids, vec![1, 3, 2, 9]);
    }

    #[test]
    fn top_k_truncates_and_handles_zero() {
        let mut idx = SmkIndex::new(1).unwrap();
        for id in 0..50 {
            idx.upsert(entry(id, vec![1.0], default_key())).unwrap();
        }
        assert_eq!(idx.query_top_k_filtered(&[1.0], 3, &SmkQuery::default()).unwrap().len(), 3);
        assert!(idx.query_top_k_filtered(&[1.0], 0, &SmkQuery::default()).unwrap().is_empty());
    }

    #[test]
    fn dimension_mismatches_are_errors_not_panics() {
        let mut idx = SmkIndex::new(2).unwrap();
        assert!(idx.upsert(entry(1, vec![1.0], default_key())).is_err());
        assert!(idx.query_top_k_filtered(&[1.0], 1, &SmkQuery::default()).is_err());
    }
}
