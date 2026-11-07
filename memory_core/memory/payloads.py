"""Bounded payload cache. Membership is resident; selected values come from SQLite."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING

from sqlalchemy import text

from .models import MEMORY_ADAPTER, Memory

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class Payloads(MutableMapping[str, Memory]):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.ids = set()
        self.resident = OrderedDict()
        self.session = None
        self.owner = None

    def bind(self, session: Session, owner: str) -> None:
        self.session, self.owner = session, owner

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __contains__(self, key: object) -> bool:
        return key in self.ids

    def __getitem__(self, key: str) -> Memory:
        if key not in self.ids:
            raise KeyError(key)
        if key in self.resident:
            self.resident.move_to_end(key)
            return self.resident[key]
        # bind() runs at the start of every snapshot, before any read.
        assert self.session is not None
        row = self.session.execute(
            text("SELECT payload FROM memories_v1 WHERE owner=:o AND id=:id"),
            {"o": self.owner, "id": key},
        ).scalar_one()
        memory = MEMORY_ADAPTER.validate_json(row)
        if self.capacity:
            self.resident[key] = memory
            while len(self.resident) > self.capacity:
                self.resident.popitem(last=False)
        return memory

    def __setitem__(self, key: str, value: Memory) -> None:
        self.ids.add(key)
        self.resident.pop(key, None)

    def __delitem__(self, key: str) -> None:
        self.ids.remove(key)
        self.resident.pop(key, None)
