"""Offline real-embedding smoke evaluation. Requires an explicitly supplied local model."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from memory_core import MemoryCore
from memory_core.indexing.semantic import IndexPolicy
from memory_core.memory.embedding import LocalSentenceEmbedder
from memory_core.memory.evaluation import RetrievalCase, evaluate_retrieval
from memory_core.memory.models import ObservationMemory, RecallRequest, RememberRequest
from memory_core.runtime import MaintenancePolicy


async def run(args):
    fixture = json.loads(Path(args.fixture).read_text())
    embedder = LocalSentenceEmbedder(
        args.model_path,
        model_id=args.model_id,
        revision=args.revision,
        dimensions=args.dimensions,
    )
    cases = [RetrievalCase.model_validate(q) for q in fixture["queries"]]
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        for backend in ("exact", "hnsw"):
            path = Path(directory) / f"{backend}.db"

            def open_memory(path=path, backend=backend):
                return MemoryCore.local(
                    path,
                    embedding_spec=embedder.spec,
                    embedder=embedder,
                    index_policy=IndexPolicy(backend=backend, exact_threshold=1),
                    maintenance=MaintenancePolicy(
                        background=False, dream_after_writes=10
                    ),
                )

            memory = open_memory()
            for item in fixture["memories"]:
                await memory.call(
                    RememberRequest(
                        memory=ObservationMemory(
                            id=item["id"],
                            summary=item["text"],
                            observation=item["text"],
                        )
                    ),
                    owner_id="evaluation",
                )
            before = await evaluate_retrieval(
                memory.service, cases, owner_id="evaluation", k=3
            )
            while await memory.worker.run_once():
                pass
            await memory.service.checkpoint_indexes(force=True)
            memory.service.store.db.engine.dispose()
            memory = open_memory()
            after = await evaluate_retrieval(
                memory.service, cases, owner_id="evaluation", k=3
            )
            unknown = []
            for question in fixture["unanswerable"]:
                response = await memory.call(
                    RecallRequest(query=question, limit=3), owner_id="evaluation"
                )
                unknown.append(
                    {
                        "query": question,
                        "candidate_count": len(response.hits),
                        "top_score": response.hits[0].score if response.hits else None,
                    }
                )
            results[backend] = {
                "before": before.model_dump(),
                "after_restart_and_dream": after.model_dump(),
                "unanswerable_candidates": unknown,
            }
            memory.service.store.db.engine.dispose()
    return {
        "model": embedder.spec.model_dump(),
        "fixture": fixture["description"],
        "results": results,
        "interpretation": "Candidates are not answers. Unanswerable queries require knowledge evaluation; similarity alone does not establish support. This run does not evaluate an LLM reconciler.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument(
        "--fixture",
        default=str(
            Path(__file__).resolve().parents[1] / "tests/fixtures/semantic_memory.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))
