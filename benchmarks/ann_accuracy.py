"""Compare the real native exact index and HNSW on identical held-out queries."""

import argparse
import json
import resource
import statistics
import time

import numpy as np
from memory_core._native import PyVectorIndex

from memory_core.indexing.semantic import HnswIndex, IndexPolicy


def run(count, dimensions, queries, ef):
    rng = np.random.default_rng(9241)
    values = rng.normal(size=(count, dimensions)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1)[:, None]
    exact = PyVectorIndex(dimensions)
    policy = IndexPolicy(backend="hnsw", ef_search=ef)
    ann = HnswIndex(dimensions, policy)
    ids = [str(i) for i in range(count)]
    started = time.perf_counter()
    for uid, vector in zip(ids, values, strict=True):
        exact.upsert(uid, vector.tolist())
        ann.upsert(uid, vector)
    build = time.perf_counter() - started
    overlap, exact_ms, ann_ms = [], [], []
    for query in rng.normal(size=(queries, dimensions)).astype(np.float32):
        query /= np.linalg.norm(query)
        started = time.perf_counter()
        expected = exact.search(query.tolist(), ids, 10)
        exact_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        found = ann.search(query, ids, 10)
        ann_ms.append((time.perf_counter() - started) * 1000)
        overlap.append(len({x[0] for x in expected} & {x[0] for x in found}) / 10)
        assert ann.last_search == "hnsw"
    manifest, blob = ann.snapshot()

    def stats(samples):
        return {
            "p50_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
        }

    return {
        "count": count,
        "dimensions": dimensions,
        "queries": queries,
        "policy": policy.model_dump(),
        "build_both_seconds": build,
        "mean_top10_overlap": statistics.mean(overlap),
        "minimum_query_overlap": min(overlap),
        "acceptance_target": 0.98,
        "accepted": statistics.mean(overlap) >= 0.98,
        "exact": stats(exact_ms),
        "hnsw": stats(ann_ms),
        "graph_bytes": len(blob),
        "manifest_bytes": len(manifest.encode()),
        "process_peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "note": "Random normalized vectors, not semantic quality. RSS includes both indexes and fixture. Independent queries are not indexed.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--ef", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(run(args.count, args.dimensions, args.queries, args.ef), indent=2))
