"""Column codecs shared by the stores: embeddings, tag sets, and JSON objects.

New embeddings use versioned little-endian binary64 values, preserving Python float
precision. Legacy JSON cells remain readable. Corrupt cells decode to "absent".
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def encode_embedding(embedding: Iterable[float] | None) -> bytes | None:
    if embedding is None:
        return None
    values = [float(x) for x in embedding]
    return b"MCV1" + struct.pack(f"<{len(values)}d", *values)


def decode_embedding(raw: bytes | None) -> list[float] | None:
    if not raw:
        return None
    try:
        if raw.startswith(b"MCV1"):
            payload = raw[4:]
            if len(payload) % 8:
                return None
            return list(struct.unpack(f"<{len(payload) // 8}d", payload))
        decoded = json.loads(raw.decode("utf-8"))
        return [float(x) for x in decoded] if isinstance(decoded, list) else None
    except (ValueError, TypeError, UnicodeDecodeError, struct.error):
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
