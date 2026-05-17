"""Unit tests for core decorators."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.core.decorators import (
    RateLimit,
    _TOOL_REGISTRY,
    get_tool,
    list_tools,
    timed,
    tool,
    with_timeout,
)


@pytest.mark.asyncio
async def test_tool_registry():
    @tool(name="test_tool_unit", description="A test tool")
    async def my_tool(x: int) -> int:
        return x * 2

    assert get_tool("test_tool_unit") is not None
    tools = list_tools()
    names = [t["name"] for t in tools]
    assert "test_tool_unit" in names


@pytest.mark.asyncio
async def test_tool_preserves_signature():
    @tool()
    async def documented_tool(value: str) -> str:
        """My docstring."""
        return value.upper()

    assert documented_tool.__name__ == "documented_tool"
    result = await documented_tool("hello")
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_timed_decorator():
    @timed
    async def slow_fn() -> str:
        await asyncio.sleep(0.01)
        return "done"

    result = await slow_fn()
    assert result == "done"


@pytest.mark.asyncio
async def test_rate_limit():
    limiter = RateLimit(calls_per_second=20.0)

    @limiter
    async def fast_fn() -> float:
        return time.monotonic()

    t1 = await fast_fn()
    t2 = await fast_fn()
    # At 20 calls/s each call is spaced >=50ms apart
    assert t2 - t1 >= 0.04


@pytest.mark.asyncio
async def test_with_timeout_success():
    @with_timeout(seconds=5.0)
    async def quick() -> str:
        return "ok"

    assert await quick() == "ok"


@pytest.mark.asyncio
async def test_with_timeout_fires():
    @with_timeout(seconds=0.05)
    async def slow() -> str:
        await asyncio.sleep(10)
        return "never"

    with pytest.raises(asyncio.TimeoutError):
        await slow()
