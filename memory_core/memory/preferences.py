"""Complete scoped preference resolution, independent of semantic top-k."""

from __future__ import annotations

import json
from collections import defaultdict

from .models import (
    PreferenceConflict,
    PreferencesResponse,
    ProjectScope,
    Status,
)


def resolve_preferences(cache, registry, generation, context):
    attributes = registry.normalize(context.attributes)
    groups = defaultdict(list)
    ids = cache.matching_preferences(attributes, context.project_id)
    for uid in sorted(ids):
        memory = cache.memories[uid]
        groups[memory.preference_key].append(memory)
    effective, conflicts = [], []
    for key, memories in sorted(groups.items()):
        # Explicit project exceptions override global rules. Within a scope,
        # ambiguity is surfaced rather than settled by similarity or recency.
        explicit = [m for m in memories if m.explicit]
        memories = explicit or memories
        project = [m for m in memories if isinstance(m.scope, ProjectScope)]
        relevant = project or memories
        signatures = {json.dumps(m.value, sort_keys=True) for m in relevant}
        if len(signatures) > 1 or any(m.status == Status.DISPUTED for m in relevant):
            conflicts.append(PreferenceConflict(preference_key=key, memories=relevant))
        else:
            effective.extend(
                relevant
            )  # retain every source rather than dropping evidence
    return PreferencesResponse(
        effective=effective, conflicts=conflicts, generation=generation
    ).model_copy(deep=True)
