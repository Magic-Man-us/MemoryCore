"""Registry validation and posting-list filtering, independent of domain vocabulary."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict

from .errors import SchemaError
from .models import (
    All,
    AnyOf,
    ContainsAll,
    ContainsAny,
    Eq,
    Exists,
    FieldDefinition,
    FieldType,
    Filter,
    MetadataValue,
    Range,
)

# Field shapes only; values (python, uv, customer-specific tools, etc.) are never predefined.
DEFAULT_FIELDS = tuple(
    FieldDefinition(name=name, type=FieldType.KEYWORDS)
    for name in ("languages", "tools", "topics", "domains", "activities", "frameworks")
) + tuple(
    FieldDefinition(name=name, type=FieldType.KEYWORD, read_only=True)
    for name in ("kind", "status")
)


class Registry:
    def __init__(self, fields: list[FieldDefinition]):
        self.fields = {f.name: f for f in fields}
        self.aliases: dict[str, str] = {}
        for field in fields:
            for alias in field.aliases:
                if alias in self.fields or alias in self.aliases:
                    raise SchemaError(f"alias already registered: {alias}")
                self.aliases[alias] = field.name

    def resolve(self, name: str) -> FieldDefinition:
        canonical = self.aliases.get(name, name)
        try:
            return self.fields[canonical]
        except KeyError as exc:
            raise SchemaError(
                f"unknown metadata field: {name}; call list_metadata_fields"
            ) from exc

    @staticmethod
    def check_value(field: FieldDefinition, value: MetadataValue) -> None:
        valid = {
            FieldType.KEYWORD: lambda: type(value) is str,
            FieldType.KEYWORDS: lambda: (
                type(value) is list and all(type(x) is str for x in value)
            ),
            FieldType.NUMBER: lambda: (
                type(value) in (int, float) and math.isfinite(value)
            ),
            FieldType.BOOLEAN: lambda: type(value) is bool,
        }[field.type]()
        if not valid:
            raise SchemaError(f"{field.name} requires {field.type}")

    def normalize(
        self, data: dict[str, MetadataValue], *, allow_read_only=False
    ) -> dict[str, MetadataValue]:
        result = {}
        for name, value in data.items():
            field = self.resolve(name)
            if field.read_only and not allow_read_only:
                raise SchemaError(f"{field.name} is a read-only system field")
            self.check_value(field, value)
            if field.name in result:
                raise SchemaError(f"duplicate field through alias: {field.name}")
            result[field.name] = (
                sorted(set(value)) if isinstance(value, list) else value
            )
        return result

    def check_filter(self, query: Filter, depth: int = 0) -> None:
        if depth > 8:
            raise SchemaError("filter nesting exceeds 8")
        if isinstance(query, (All, AnyOf)):
            for child in query.conditions:
                self.check_filter(child, depth + 1)
            return
        field = self.resolve(query.field)
        if isinstance(query, Exists):
            return
        if isinstance(query, (ContainsAll, ContainsAny)):
            if field.type != FieldType.KEYWORDS:
                raise SchemaError("contains operators require a keywords field")
        elif isinstance(query, Range):
            if field.type != FieldType.NUMBER:
                raise SchemaError("range requires a number field")
        else:
            if field.type == FieldType.KEYWORDS:
                raise SchemaError("use contains_any/contains_all for a keywords field")
            for value in [query.value] if isinstance(query, Eq) else query.values:
                self.check_value(field, value)


def value_key(value):
    # bool and int must never share an index key (True == 1 in Python).
    return ("number" if type(value) in (int, float) else type(value).__name__, value)


class MetadataIndex:
    def __init__(self):
        self.postings: dict[str, dict[tuple, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.present: dict[str, set[str]] = defaultdict(set)
        self.values: dict[str, dict[str, MetadataValue]] = {}
        self.numeric_sorted: dict[str, list[float]] = {}

    def remove(self, uid: str):
        previous = self.values.pop(uid, {})
        for field, value in previous.items():
            self.numeric_sorted.pop(field, None)
            self.present[field].discard(uid)
            for item in value if isinstance(value, list) else [value]:
                key = value_key(item)
                posting = self.postings[field][key]
                posting.discard(uid)
                if not posting:
                    del self.postings[field][key]

    def upsert(self, uid: str, data: dict[str, MetadataValue]):
        self.remove(uid)
        self.values[uid] = data
        for field, value in data.items():
            self.numeric_sorted.pop(field, None)
            self.present[field].add(uid)
            for item in value if isinstance(value, list) else [value]:
                self.postings[field][value_key(item)].add(uid)

    def select(self, query: Filter, registry: Registry, universe: set[str]) -> set[str]:
        registry.check_filter(query)
        return self._select(query, registry, universe)

    def _select(
        self, query: Filter, registry: Registry, universe: set[str]
    ) -> set[str]:
        if isinstance(query, (All, AnyOf)):
            parts = [self._select(q, registry, universe) for q in query.conditions]
            if isinstance(query, All):
                return set.intersection(universe, *parts)
            return set.union(set(), *parts) & universe
        field = registry.resolve(query.field).name
        if isinstance(query, Exists):
            return self.present.get(field, set()) & universe
        posting = self.postings.get(field, {})
        if isinstance(query, Range):
            if field not in self.numeric_sorted:
                self.numeric_sorted[field] = sorted(key[1] for key in posting)
            ordered = self.numeric_sorted[field]
            lo = 0 if query.minimum is None else bisect_left(ordered, query.minimum)
            hi = (
                len(ordered)
                if query.maximum is None
                else bisect_right(ordered, query.maximum)
            )
            return (
                set.union(set(), *(posting[value_key(v)] for v in ordered[lo:hi]))
                & universe
            )
        values = [query.value] if isinstance(query, Eq) else query.values
        parts = [posting.get(value_key(v), set()) for v in values]
        if isinstance(query, ContainsAll):
            # Empty contains_all matches present arrays, including empty arrays, not missing fields.
            return set.intersection(self.present.get(field, set()) & universe, *parts)
        return set.union(set(), *parts) & universe
