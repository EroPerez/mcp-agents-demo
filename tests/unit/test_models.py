"""Unit tests for Pydantic v2 domain models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.core.models import (
    AgentMessage,
    CacheKey,
    CoverageAnalysis,
    ShiftQuery,
    ToolCallRequest,
    ToolUseContent,
)


class TestToolCallRequest:
    def test_valid(self):
        req = ToolCallRequest(
            tool_name="search_shifts",
            agency_id=42,
        )
        assert req.tool_name == "search_shifts"
        assert req.agency_id == 42

    def test_invalid_tool_name(self):
        with pytest.raises(ValidationError, match="Unknown tool"):
            ToolCallRequest(tool_name="drop_table", agency_id=1)

    def test_update_requires_arguments(self):
        with pytest.raises(ValidationError, match="requires non-empty arguments"):
            ToolCallRequest(tool_name="update_shift_status", agency_id=1, arguments={})

    def test_strips_whitespace(self):
        req = ToolCallRequest(tool_name="  search_shifts  ", agency_id=1)
        assert req.tool_name == "search_shifts"


class TestShiftQuery:
    def test_valid(self):
        q = ShiftQuery(agency_id=1, date_from="2025-01-01", date_to="2025-01-31")
        assert q.agency_id == 1

    def test_date_order_validation(self):
        with pytest.raises(ValidationError, match="date_from must be"):
            ShiftQuery(agency_id=1, date_from="2025-02-01", date_to="2025-01-01")

    def test_invalid_date_format(self):
        with pytest.raises(ValidationError):
            ShiftQuery(agency_id=1, date_from="01-01-2025", date_to="2025-01-31")

    def test_json_schema_exported(self):
        schema = ShiftQuery.model_json_schema()
        assert "properties" in schema
        assert "agency_id" in schema["properties"]


class TestDiscriminatedUnion:
    def test_tool_use_content(self):
        msg = AgentMessage.model_validate({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "abc123", "tool_name": "search_shifts", "input": {}}
            ],
        })
        content = msg.content[0]
        assert isinstance(content, ToolUseContent)
        assert content.tool_name == "search_shifts"

    def test_mixed_content(self):
        msg = AgentMessage.model_validate({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Searching..."},
                {"type": "tool_use", "id": "x", "tool_name": "get_schedule", "input": {}},
            ],
        })
        assert len(msg.content) == 2


class TestCacheKey:
    def test_hashable(self):
        k1 = CacheKey(agency_id=1, date="2025-01-01", tool_name="search_shifts")
        k2 = CacheKey(agency_id=1, date="2025-01-01", tool_name="search_shifts")
        assert k1 == k2
        assert hash(k1) == hash(k2)
        cache = {k1: "result"}
        assert cache[k2] == "result"

    def test_str_repr(self):
        k = CacheKey(agency_id=42, date="2025-06-01", tool_name="get_schedule")
        assert str(k) == "get_schedule:42:2025-06-01"
