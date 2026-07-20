"""SMK feature extraction and quantization."""

from __future__ import annotations

import pytest

from tests.conftest import make_assistant_trace
from memory_core.types.smk_features import _quantize_level, build_smk_features
from memory_core.types.smk_types import Level2Bits, MemoryKind, ToolFlag, TopicBucket


class TestQuantization:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, Level2Bits.LOW),
            (0.24999, Level2Bits.LOW),
            (0.25, Level2Bits.MEDIUM),
            (0.49999, Level2Bits.MEDIUM),
            (0.5, Level2Bits.HIGH),
            (0.74999, Level2Bits.HIGH),
            (0.75, Level2Bits.EXTREME),
            (1.0, Level2Bits.EXTREME),
        ],
    )
    def test_bucket_boundaries(self, value: float, expected: Level2Bits):
        assert _quantize_level(value) is expected


class TestBuildSmkFeatures:
    def test_tool_mask_is_the_or_of_flags(self):
        features = build_smk_features(make_assistant_trace())
        assert features.tool_mask == int(ToolFlag.PY | ToolFlag.MATURIN)

    def test_no_tools_gives_empty_mask(self):
        features = build_smk_features(make_assistant_trace(tools=set()))
        assert features.tool_mask == 0

    def test_scalars_map_to_the_right_fields(self):
        features = build_smk_features(
            make_assistant_trace(
                before_state_confusion=0.8,  # difficulty EXTREME
                generality=0.3,  # MEDIUM
                importance=0.6,  # HIGH
            )
        )
        assert features.difficulty is Level2Bits.EXTREME
        assert features.generality is Level2Bits.MEDIUM
        assert features.importance is Level2Bits.HIGH
        assert features.topic is TopicBucket.RUST_PYTHON_TOOLCHAIN
        assert features.kind is MemoryKind.PATTERN

    def test_tool_flags_compose_as_bitfield(self):
        assert int(ToolFlag.RS | ToolFlag.CFN) == (1 << 0) | (1 << 4)
