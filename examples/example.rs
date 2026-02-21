
fn main() {
    // Dimension of our toy embeddings
    let dim = 64usize;
    let n = 10_000usize;
    let k = 5usize;

    let mut idx = RustMemoryIndex::new(dim);
    let mut rng = rand::rng();

    // --- Populate index ---------------------------------------------------
    // We'll create a bunch of random traces, but ensure some are clearly
    // Rust+Python+uv toolchain patterns with high generality/importance.

    let mut next_id: u64 = 0;

    // 1) Insert some high-quality Rust+Python+uv patterns (what we WANT back).
    for _ in 0..20 {
        let mut emb = Vec::with_capacity(dim);
        for _ in 0..dim {
            emb.push(rng.random::<f32>());
        }

        let smk = StructuredMemoryKey::new(
            TopicBucket::RustPythonToolchain,
            MemoryKind::Pattern,
            TOOL_RS | TOOL_PY | TOOL_UV | TOOL_MATURIN,
            Level2Bits::High,      // difficulty / novelty
            Level2Bits::High,      // generality
            Level2Bits::High,      // importance
        );

        idx.add(MemoryTrace { id: next_id, embedding: emb, smk });
        next_id += 1;
    }

    // 2) Insert some other mixed memories (noise from this query's perspective).
    for _ in 0..(n - 20) {
        let mut emb = Vec::with_capacity(dim);
        for _ in 0..dim {
            emb.push(rng.random::<f32>());
        }

        // Randomly assign topic / kind / tools.
        let topic = match rng.random::<u8>() % 4 {
            0 => TopicBucket::RustPythonToolchain,
            1 => TopicBucket::MemoryArchitecture,
            2 => TopicBucket::AwsIam,
            _ => TopicBucket::DbSchema,
        };

        let kind = match rng.random::<u8>() % 5 {
            0 => MemoryKind::Insight,
            1 => MemoryKind::Pattern,
            2 => MemoryKind::AntiPattern,
            3 => MemoryKind::Principle,
            _ => MemoryKind::Workflow,
        };

        let mut tools: u16 = 0;
        if rng.random::<f32>() < 0.4 {
            tools |= TOOL_RS;
        }
        if rng.random::<f32>() < 0.4 {
            tools |= TOOL_PY;
        }
        if rng.random::<f32>() < 0.2 {
            tools |= TOOL_UV;
        }
        if rng.random::<f32>() < 0.2 {
            tools |= TOOL_MATURIN;
        }
        if rng.random::<f32>() < 0.2 {
            tools |= TOOL_CFN;
        }

        let diff = match rng.random::<u8>() % 4 {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        };
        let gen_level = match rng.random::<u8>() % 4 {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        };
        let imp = match rng.random::<u8>() % 4 {
            0 => Level2Bits::Low,
            1 => Level2Bits::Medium,
            2 => Level2Bits::High,
            _ => Level2Bits::Extreme,
        };

        let smk = StructuredMemoryKey::new(topic, kind, tools, diff, gen_level, imp);
        idx.add(MemoryTrace { id: next_id, embedding: emb, smk });
        next_id += 1;
    }

    // Build a random query embedding.
    let mut query = Vec::with_capacity(dim);
    for _ in 0..dim {
        query.push(rng.random::<f32>());
    }

    // --- SMK-aware query: Rust+Python+uv toolchain patterns ----------------

    let smk_query = SmkQuery {
        topic: Some(TopicBucket::RustPythonToolchain),
        required_tools_mask: TOOL_RS | TOOL_PY | TOOL_UV,
        allowed_kinds: Some(vec![MemoryKind::Pattern, MemoryKind::AntiPattern]),
        min_generality: Some(Level2Bits::High),
        min_importance: Some(Level2Bits::High),
    };

    let t0 = Instant::now();
    let results = idx.query_top_k_filtered(&query, k, &smk_query);
    let elapsed = t0.elapsed();

    println!(
        "Rust+SMK: n={}, dim={}, k={}, took {:?} ({} ms)",
        n,
        dim,
        k,
        elapsed,
        elapsed.as_secs_f64() * 1000.0
    );

    println!("Top results (id, score, SMK):");
    for (id, score, smk) in &results {
        println!("  id={}  score={:.4}  {}", id, score, smk);
    }
}


