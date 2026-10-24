"""Column codecs shared by the stores: embeddings, tag sets, and JSON objects.

Embeddings are stored as JSON-encoded float lists in a binary column, the format
every existing database already holds. Decoders are lenient: a corrupt cell decodes
to "absent" rather than failing a whole hydration.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def encode_embedding(embedding: Iterable[float] | None) -> bytes | None:
    if embedding is None:
        return None
    return json.dumps([float(x) for x in embedding]).encode("utf-8")


def decode_embedding(raw: bytes | None) -> list[float] | None:
    if not raw:
        return None
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(decoded, list):
        return [float(x) for x in decoded]
    return None


def decode_tags(raw: str | None) -> set[str]:
    if not raw:
        return set()
    try:
        tags = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    return set(tags) if isinstance(tags, list) else set()


def decode_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
