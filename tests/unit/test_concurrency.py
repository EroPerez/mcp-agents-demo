"""Unit tests for concurrency utilities."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.core.concurrency import (
    gather_with_limit,
    run_in_thread,
    run_pipeline,
)


@pytest.mark.asyncio
async def test_gather_with_limit_ordering():
    """Results should have same length as inputs."""
    async def echo(x: int) -> int:
        await asyncio.sleep(0.01)
        return x

    results = await gather_with_limit([echo(i) for i in range(10)], max_concurrent=3)
    assert len(results) == 10
    assert set(results) == set(range(10))


@pytest.mark.asyncio
async def test_gather_with_limit_caps_concurrency():
    """At most max_concurrent coroutines should run simultaneously."""
    active = 0
    peak = 0

    async def track(_: int) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1

    await gather_with_limit([track(i) for i in range(20)], max_concurrent=4)
    assert peak <= 4


@pytest.mark.asyncio
async def test_run_pipeline_processes_all():
    results = []

    async def process(item: int) -> int:
        await asyncio.sleep(0.01)
        return item * 2

    results = await run_pipeline(list(range(8)), process, n_workers=4)
    assert len(results) == 8
    assert set(results) == {i * 2 for i in range(8)}


@pytest.mark.asyncio
async def test_run_in_thread_doesnt_block_event_loop():
    """Blocking function must not starve async tasks."""

    def slow_sync(n: float) -> float:
        time.sleep(n)
        return n

    # Run both in parallel — if thread blocks loop, gather takes 2× longer
    start = time.perf_counter()
    await asyncio.gather(
        run_in_thread(slow_sync, 0.1),
        run_in_thread(slow_sync, 0.1),
    )
    elapsed = time.perf_counter() - start
    # Should complete in ~0.1s, definitely not 0.2s
    assert elapsed < 0.18


@pytest.mark.asyncio
async def test_run_in_thread_propagates_context():
    from contextvars import ContextVar

    my_var: ContextVar[str] = ContextVar("my_var", default="")
    my_var.set("hello")

    def read_var() -> str:
        return my_var.get()

    result = await run_in_thread(read_var)
    assert result == "hello"
