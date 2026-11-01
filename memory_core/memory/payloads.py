"""Bounded payload cache. Membership is resident; selected values come from SQLite."""

from collections import OrderedDict
from collections.abc import MutableMapping

from sqlalchemy import text

from .models import MEMORY_ADAPTER


class Payloads(MutableMapping):
    def __init__(self, capacity):
        self.capacity = capacity
        self.ids = set()
        self.resident = OrderedDict()
        self.session = None
        self.owner = None

    def bind(self, session, owner):
        self.session, self.owner = session, owner

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __contains__(self, key):
        return key in self.ids

    def __getitem__(self, key):
        if key not in self.ids:
            raise KeyError(key)
        if key in self.resident:
            self.resident.move_to_end(key)
            return self.resident[key]
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

    def __setitem__(self, key, value):
        self.ids.add(key)
        self.resident.pop(key, None)

    def __delitem__(self, key):
        self.ids.remove(key)
        self.resident.pop(key, None)
