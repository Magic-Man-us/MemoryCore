"""Real HNSW adapter and SQLite checkpoint/recovery regressions."""

import pytest
from sqlalchemy import text

from memory_core import EmbeddingSpec, MemoryCore
from memory_core.indexing.semantic import HnswIndex, IndexPolicy
from memory_core.memory.models import (
    ObservationMemory,
    PreferenceMemory,
    RecallRequest,
    RememberRequest,
    TaskContext,
)
from memory_core.runtime import MaintenancePolicy

np = pytest.importorskip("numpy")
pytest.importorskip("hnswlib")


def test_hnsw_matches_exact_and_enforces_selective_scope():
    rng = np.random.default_rng(71)
    vectors = rng.normal(size=(3000, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    index = HnswIndex(
        32, IndexPolicy(backend="hnsw", exact_threshold=100, ef_search=256)
    )
    for i, v in enumerate(vectors):
        index.upsert(str(i), v)
    overlap = []
    for q in rng.normal(size=(20, 32)).astype(np.float32):
        q /= np.linalg.norm(q)
        expected = {str(i) for i in np.argsort(-(vectors @ q))[:10]}
        found = {uid for uid, _ in index.search(q, list(map(str, range(3000))), 10)}
        overlap.append(len(found & expected) / 10)
        assert index.last_search == "hnsw"
    assert np.mean(overlap) >= 0.98
    assert {uid for uid, _ in index.search(vectors[0], ["1", "2", "3"], 10)} == {
        "1",
        "2",
        "3",
    }
    assert index.last_search == "exact"
    index.remove("0")
    assert "0" not in dict(index.search(vectors[0], list(map(str, range(3000))), 10))
    index.upsert("0", vectors[0])
    assert index.search(vectors[0], ["0"], 1)[0][0] == "0"
    manifest, blob = index.snapshot()
    restored = HnswIndex(32, index.policy)
    restored.restore(manifest, blob)
    assert restored.search(vectors[0], ["0"], 1)[0][0] == "0"
    with pytest.raises(ValueError):
        restored.restore(manifest, blob + b"corrupt")


class Embedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def open_memory(path):
    return MemoryCore.local(
        path,
        embedding_spec=EmbeddingSpec(model_id="test", revision="1", dimensions=2),
        embedder=Embedder(),
        index_policy=IndexPolicy(backend="hnsw", payload_cache_size=3),
        maintenance=MaintenancePolicy(background=False),
    )


async def test_checkpoint_replays_unsaved_writes_and_corruption_rebuilds(tmp_path):
    path = tmp_path / "memory.db"
    memory = open_memory(path)
    for i in range(20):
        await memory.call(
            RememberRequest(
                memory=ObservationMemory(
                    id=str(i), summary=f"Fact {i}", observation=f"Fact {i}"
                )
            ),
            owner_id="alice",
        )
    await memory.service.checkpoint_indexes(force=True)
    await memory.call(
        RememberRequest(
            memory=ObservationMemory(
                id="new", summary="New fact", observation="New fact"
            )
        ),
        owner_id="alice",
    )
    memory.service.store.db.engine.dispose()
    memory = open_memory(path)
    result = await memory.call(RecallRequest(query="New"), owner_id="alice")
    assert "new" in {h.memory.id for h in result.hits}
    cache = memory.service._caches["alice"]
    assert len(cache.memories) == 21 and len(cache.memories.resident) <= 3
    assert not (await memory.call(RecallRequest(query="New"), owner_id="bob")).hits
    with memory.queue.transaction("alice") as s:
        s.execute(
            text("UPDATE memory_vector_snapshots SET graph=:b WHERE owner='alice'"),
            {"b": b"broken"},
        )
    memory.service.store.db.engine.dispose()
    memory = open_memory(path)
    assert "new" in {
        h.memory.id
        for h in (await memory.call(RecallRequest(query="New"), owner_id="alice")).hits
    }
    memory.service.store.db.engine.dispose()


async def test_mandatory_preferences_remain_exact_with_hnsw(tmp_path):
    memory = open_memory(tmp_path / "prefs.db")
    for i in range(15):
        await memory.call(
            RememberRequest(
                memory=PreferenceMemory(
                    summary=f"Rule {i}",
                    preference_key=f"rule_{i}",
                    value=True,
                    applicability={"languages": ["python"]},
                )
            ),
            owner_id="alice",
        )
    prefs = await memory.service.resolve_prefs(
        TaskContext(task_id="task", attributes={"languages": ["python"]}),
        owner_id="alice",
    )
    assert prefs.complete and len(prefs.effective) == 15
    memory.service.store.db.engine.dispose()


def test_filtered_fallback_and_checkpoint_reclaims_deleted_nodes():
    index = HnswIndex(
        2, IndexPolicy(backend="hnsw", exact_threshold=1, max_candidates=2)
    )
    for i in range(100):
        index.upsert(str(i), [1.0, i / 100])
    assert len(index.search([1.0, 0.0], [str(i) for i in range(90, 100)], 2)) == 2
    assert index.last_search == "filtered_exact"
    for i in range(80):
        index.remove(str(i))
    manifest, blob = index.snapshot()
    assert len(index.labels) == 20
    restored = HnswIndex(2, index.policy)
    restored.restore(manifest, blob)
    assert restored.active == {str(i) for i in range(80, 100)}
    restored.upsert("new", [1.0, 0.0])
    assert restored.search([1.0, 0.0], ["new"], 1)[0][0] == "new"
