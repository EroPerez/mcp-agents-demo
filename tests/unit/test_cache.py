"""Unit tests for the cache layer (in-memory backend)."""

from __future__ import annotations

import pytest

from src.core.cache import InMemoryCache, _make_key, reset_cache


def test_make_key_deterministic():
    k1 = _make_key("ns", {"a": 1, "b": 2})
    k2 = _make_key("ns", {"b": 2, "a": 1})  # different order
    assert k1 == k2                           # same result (sort_keys=True)


def test_make_key_different_namespaces():
    k1 = _make_key("search_shifts",    {"agency_id": 1})
    k2 = _make_key("analyze_coverage", {"agency_id": 1})
    assert k1 != k2


@pytest.mark.asyncio
async def test_miss_then_hit():
    cache = InMemoryCache()
    payload = {"agency_id": 42, "date": "2025-01-01"}

    assert await cache.get("test_ns", payload) is None

    await cache.set("test_ns", payload, {"result": "ok"})
    hit = await cache.get("test_ns", payload)
    assert hit == {"result": "ok"}


@pytest.mark.asyncio
async def test_ttl_expiry():
    import asyncio
    import time

    cache = InMemoryCache()
    payload = {"x": 1}
    # Set with a past expiry directly
    key = _make_key("ns", payload)
    cache._store[key] = ("value", time.monotonic() - 1)  # already expired
    assert await cache.get("ns", payload) is None


@pytest.mark.asyncio
async def test_delete():
    cache = InMemoryCache()
    payload = {"y": 2}
    await cache.set("ns", payload, "to-delete", ttl=60)
    await cache.delete("ns", payload)
    assert await cache.get("ns", payload) is None


@pytest.mark.asyncio
async def test_flush():
    cache = InMemoryCache()
    for i in range(5):
        await cache.set("ns", {"i": i}, i, ttl=60)
    await cache.flush()
    for i in range(5):
        assert await cache.get("ns", {"i": i}) is None


@pytest.mark.asyncio
async def test_get_cache_returns_in_memory(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "")
    reset_cache()
    from src.core.cache import get_cache
    cache = await get_cache()
    assert isinstance(cache, InMemoryCache)
    reset_cache()
