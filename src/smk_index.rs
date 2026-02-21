
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct StructuredMemoryKey(pub(crate) u64);

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum TopicBucket {
    RustPythonToolchain = 1,
    MemoryArchitecture = 2,
    AwsIam = 3,
    DbSchema = 4,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Level2Bits {
    Low = 0,
    Medium = 1,
    High = 2,
    Extreme = 3,
}

// Tool bitflags (16 bits max).
pub const TOOL_RS: u16 = 1 << 0;
pub const TOOL_PY: u16 = 1 << 1;
pub const TOOL_UV: u16 = 1 << 2;
pub const TOOL_MATURIN: u16 = 1 << 3;
pub const TOOL_CFN: u16 = 1 << 4;

impl StructuredMemoryKey {
    pub fn new(
        topic: TopicBucket,
        kind: MemoryKind,
        tool_mask: u16,
        difficulty: Level2Bits,
        generality: Level2Bits,
        importance: Level2Bits,
    ) -> Self {
        let mut v: u64 = 0;

        let topic_bits = topic as u64 & 0xFF;
        let kind_bits = kind as u64 & 0x7;
        let tool_bits = tool_mask as u64 & 0xFFFF;
        let diff_bits = difficulty as u64 & 0x3;
        let gen_bits = generality as u64 & 0x3;
        let imp_bits = importance as u64 & 0x3;

        v |= topic_bits;            // bits 0..=7
        v |= kind_bits << 8;        // bits 8..=10
        v |= tool_bits << 11;       // bits 11..=26
        v |= diff_bits << 27;       // bits 27..=28
        v |= gen_bits << 29;        // bits 29..=30
        v |= imp_bits << 31;        // bits 31..=32

        StructuredMemoryKey(v)
    }

    pub fn raw(&self) -> u64 {
        self.0
    }

    pub fn topic(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    pub fn kind(&self) -> MemoryKind {
        let k = ((self.0 >> 8) & 0x7) as u8;
        match k {
            0 => MemoryKind::Insight,
            1 => MemoryKind::Pattern,
            2 => MemoryKind::AntiPattern,
            3 => MemoryKind::Principle,
            4 => MemoryKind::Workflow,
            _ => MemoryKind::Insight, // fallback, should not happen
        }
    }

    pub fn tool_mask(&self) -> u16 {
        ((self.0 >> 11) & 0xFFFF) as u16
    }

    pub fn difficulty(&self) -> Level2Bits {
        let v = ((self.0 >> 27) & 0x3) as u8;
        match v {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        }
    }

    pub fn generality(&self) -> Level2Bits {
        let v = ((self.0 >> 29) & 0x3) as u8;
        match v {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        }
    }

    pub fn importance(&self) -> Level2Bits {
        let v = ((self.0 >> 31) & 0x3) as u8;
        match v {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        }
    }
}

impl fmt::Display for StructuredMemoryKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let topic_str = match self.topic() {
            x if x == TopicBucket::RustPythonToolchain as u8 => "RUST+PYTHON_TOOLCHAIN",
            x if x == TopicBucket::MemoryArchitecture as u8 => "MEMORY_ARCH",
            x if x == TopicBucket::AwsIam as u8 => "AWS_IAM",
            x if x == TopicBucket::DbSchema as u8 => "DB_SCHEMA",
            _ => "UNKNOWN_TOPIC",
        };

        let kind_str = match self.kind() {
            MemoryKind::Insight => "INS",
            MemoryKind::Pattern => "PAT",
            MemoryKind::AntiPattern => "ANTI",
            MemoryKind::Principle => "PRIN",
            MemoryKind::Workflow => "WF",
        };

        let mut tools = Vec::new();
        let mask = self.tool_mask();
        if mask & TOOL_RS != 0 {
            tools.push("RS");
        }
        if mask & TOOL_PY != 0 {
            tools.push("PY");
        }
        if mask & TOOL_UV != 0 {
            tools.push("UV");
        }
        if mask & TOOL_MATURIN != 0 {
            tools.push("MATURIN");
        }
        if mask & TOOL_CFN != 0 {
            tools.push("CFN");
        }
        let tools_str = if tools.is_empty() {
            "NONE".to_string()
        } else {
            tools.join("+")
        };

        let lvl_str = |lvl: Level2Bits| match lvl {
            Level2Bits::Low => "LO",
            Level2Bits::Medium => "MD",
            Level2Bits::High => "HI",
            Level2Bits::Extreme => "XHI",
        };

        write!(
            f,
            "SMK:V1: T-{}+TO-{}+K-{}+G-{}+D-{}+I-{}",
            topic_str,
            tools_str,
            kind_str,
            lvl_str(self.generality()),
            lvl_str(self.difficulty()),
            lvl_str(self.importance()),
        )
    }
}

/// Query-time filter that uses the SMK to prune candidates before cosine.
#[derive(Debug, Clone)]
pub struct SmkQuery {
    pub topic: Option<TopicBucket>,
    pub required_tools_mask: u16,
    pub allowed_kinds: Option<Vec<MemoryKind>>,
    pub min_generality: Option<Level2Bits>,
    pub min_importance: Option<Level2Bits>,
}

impl SmkQuery {
    pub fn matches(&self, smk: &StructuredMemoryKey) -> bool {
        if let Some(t) = self.topic {
            if smk.topic() != t as u8 {
                return false;
            }
        }

        if self.required_tools_mask != 0 {
            if (smk.tool_mask() & self.required_tools_mask) != self.required_tools_mask {
                return false;
            }
        }

        if let Some(ref kinds) = self.allowed_kinds {
            let k = smk.kind();
            if !kinds.contains(&k) {
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

/// --- Index + cosine similarity -------------------------------------------

#[derive(Debug, Clone)]
pub struct MemoryTrace {
    pub id: u64,
    pub embedding: Vec<f32>,
    pub smk: StructuredMemoryKey,
}

pub struct RustMemoryIndex {
    pub dim: usize,
    pub memories: Vec<MemoryTrace>,
}

impl RustMemoryIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, memories: Vec::new() }
    }

    pub fn add(&mut self, mem: MemoryTrace) {
        assert_eq!(mem.embedding.len(), self.dim);
        self.memories.push(mem);
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            let x = a[i];
            let y = b[i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = (norm_a.sqrt() * norm_b.sqrt()) + 1e-9;
        dot / denom
    }

    pub fn query_top_k_filtered(
        &self,
        query: &[f32],
        k: usize,
        smk_query: &SmkQuery,
    ) -> Vec<(u64, f32, StructuredMemoryKey)> {
        assert_eq!(query.len(), self.dim);

        let mut scores: Vec<(u64, f32, StructuredMemoryKey)> = self
            .memories
            .iter()
            .filter(|m| smk_query.matches(&m.smk))
            .map(|m| {
                let s = Self::cosine_similarity(&m.embedding, query);
                (m.id, s, m.smk)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    }
}
