"""Column codecs shared by the stores: embeddings and JSON values.

New embeddings use versioned little-endian binary64 values, preserving Python float
precision. JSON embedding cells written before the binary encoding remain readable.
Corrupt cells decode to "absent".
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING

from pydantic import JsonValue, TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Iterable

JSON_VALUE = TypeAdapter(JsonValue)
_JSON_EMBEDDING = TypeAdapter(list[float])


def canonical_json(value: JsonValue) -> str:
    """Key-sorted JSON for identity comparison and event-replay checks.

    Pydantic serialization cannot sort keys, and stored replay payloads were written in
    exactly this form, so these bytes come from the standard library.
    """
    return json.dumps(value, sort_keys=True)


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
        return _JSON_EMBEDDING.validate_json(raw)
    except (ValueError, struct.error):
        return None
