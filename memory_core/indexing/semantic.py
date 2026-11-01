"""Local candidate indexes. SQLite owns records; graph snapshots are disposable."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class IndexPolicy(BaseModel):
    backend: Literal["exact", "hnsw"] = "exact"
    exact_threshold: int = Field(default=1000, ge=1)
    ef_search: int = Field(default=256, ge=1)
    ef_construction: int = Field(default=200, ge=1)
    connections: int = Field(default=24, ge=2, le=128)
    max_candidates: int = Field(default=4096, ge=1)
    payload_cache_size: int = Field(default=256, ge=0)
    checkpoint_after_updates: int = Field(default=1000, ge=1)


class CandidateIndex(Protocol):
    def upsert(self, uid: str, vector: list[float]) -> None: ...
    def remove(self, uid: str) -> bool: ...
    def search(
        self, vector: list[float], ids: list[str], k: int
    ) -> list[tuple[str, float]]: ...


class HnswIndex:
    def __init__(self, dimensions: int, policy: IndexPolicy):
        try:
            import hnswlib
            import numpy as np
        except ImportError as exc:
            raise ImportError("HNSW requires `uv sync --extra ann`") from exc
        self.np, self.library = np, hnswlib
        self.dimensions, self.policy = dimensions, policy
        self.lock = RLock()
        self.labels = {}
        self.reverse = {}
        self.active = set()
        self.capacity = 1024
        self.graph = self._new(self.capacity)
        self.updates = 0
        self.last_search = "none"
        self.last_candidates = 0

    def _new(self, capacity):
        graph = self.library.Index(space="cosine", dim=self.dimensions)
        graph.init_index(
            max_elements=capacity,
            M=self.policy.connections,
            ef_construction=self.policy.ef_construction,
            random_seed=42,
        )
        graph.set_num_threads(1)
        graph.set_ef(self.policy.ef_search)
        return graph

    def _vector(self, vector):
        v = self.np.asarray(vector, dtype=self.np.float32)
        if v.shape != (self.dimensions,) or not self.np.isfinite(v).all():
            raise ValueError("invalid embedding dimensions or values")
        norm = self.np.linalg.norm(v.astype(self.np.float64))
        if not norm:
            raise ValueError("zero embedding is not searchable")
        return (v / norm).astype(self.np.float32)

    def upsert(self, uid, vector):
        v = self._vector(vector)
        with self.lock:
            if uid not in self.labels:
                if len(self.labels) >= self.capacity:
                    self.capacity *= 2
                    self.graph.resize_index(self.capacity)
                self.labels[uid] = len(self.labels)
                self.reverse[self.labels[uid]] = uid
            self.graph.add_items(v.reshape(1, -1), [self.labels[uid]])
            self.active.add(uid)
            self.updates += 1

    def remove(self, uid):
        with self.lock:
            if uid not in self.active:
                return False
            self.graph.mark_deleted(self.labels[uid])
            self.active.remove(uid)
            self.updates += 1
            return True

    def _exact(self, vector, ids, k):
        # Bound the scratch matrix; do not materialize all eligible vectors at once.
        best = []
        for start in range(0, len(ids), 1024):
            batch = ids[start : start + 1024]
            values = self.graph.get_items([self.labels[uid] for uid in batch]) @ vector
            best.extend(zip(batch, map(float, values), strict=True))
            best = sorted(best, key=lambda hit: (-hit[1], hit[0]))[:k]
        return best

    def search(self, vector, ids, k):
        query = self._vector(vector)
        with self.lock:
            eligible = self.active.intersection(ids)
            k = min(k, len(eligible))
            if not k:
                self.last_search, self.last_candidates = "empty", 0
                return []
            if len(eligible) <= max(self.policy.exact_threshold, k):
                self.last_search, self.last_candidates = "exact", len(eligible)
                return self._exact(query, sorted(eligible), k)
            reverse = self.reverse
            count = min(len(self.active), max(k * 4, self.policy.ef_search))
            ceiling = min(len(self.active), max(k, self.policy.max_candidates))
            while True:
                count = min(count, ceiling)
                self.graph.set_ef(max(self.policy.ef_search, count))
                labels, distances = self.graph.knn_query(query, k=count)
                found = [
                    (reverse[int(label)], 1.0 - float(distance))
                    for label, distance in zip(labels[0], distances[0], strict=True)
                    if reverse[int(label)] in eligible
                ]
                if len(found) >= k:
                    self.last_search, self.last_candidates = "hnsw", count
                    return sorted(found, key=lambda hit: (-hit[1], hit[0]))[:k]
                if count >= ceiling:
                    self.last_search, self.last_candidates = (
                        "filtered_exact",
                        len(eligible),
                    )
                    return self._exact(query, sorted(eligible), k)
                count *= 2

    def snapshot(self):
        with self.lock:
            # Rebuild tombstones at checkpoint, not during foreground queries.
            if len(self.labels) - len(self.active) > max(32, len(self.labels) // 5):
                old, mapping = self.graph, self.labels
                self.capacity = max(1024, len(self.active) * 2)
                self.graph, self.labels = self._new(self.capacity), {}
                ids = sorted(self.active)
                for start in range(0, len(ids), 256):
                    batch = ids[start : start + 256]
                    values = old.get_items([mapping[uid] for uid in batch])
                    labels = list(range(start, start + len(batch)))
                    self.graph.add_items(values, labels)
                    self.labels.update(zip(batch, labels, strict=True))
            self.reverse = {label: uid for uid, label in self.labels.items()}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "index.bin"
                self.graph.save_index(str(path))
                blob = path.read_bytes()
            manifest = json.dumps(
                {
                    "version": 1,
                    "labels": self.labels,
                    "active": sorted(self.active),
                    "capacity": self.capacity,
                    "digest": hashlib.sha256(blob).hexdigest(),
                }
            )
            return manifest, blob

    def restore(self, manifest, blob):
        with self.lock:
            data = json.loads(manifest)
            if (
                data["version"] != 1
                or hashlib.sha256(blob).hexdigest() != data["digest"]
            ):
                raise ValueError("invalid graph snapshot")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "index.bin"
                path.write_bytes(blob)
                graph = self.library.Index(space="cosine", dim=self.dimensions)
                graph.load_index(str(path), max_elements=data["capacity"])
            graph.set_num_threads(1)
            graph.set_ef(self.policy.ef_search)
            if set(graph.get_ids_list()) != set(data["labels"].values()):
                raise ValueError("snapshot labels do not match graph")
            self.graph, self.capacity = graph, data["capacity"]
            self.labels, self.active = data["labels"], set(data["active"])
            self.reverse = {label: uid for uid, label in self.labels.items()}
            self.updates = 0
