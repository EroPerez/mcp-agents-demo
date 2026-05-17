"""Redis-backed semantic cache for LLM responses.

Strategy:
- Exact-match cache: identical inputs → instant return (no LLM call)
- TTL-based expiry per tool/agent type
- In-memory dict fallback when Redis is unavailable (dev/test)
- CacheKey dataclass (slots=True) as canonical key format

Usage:
    cache = get_cache()
    hit = await cache.get("search_shifts", payload)
    if hit:
        return hit
    result = await llm_call(payload)
    await cache.set("search_shifts", payload, result, ttl=300)
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

from src.core.config import get_settings

log = structlog.get_logger(__name__)

# TTL per tool (seconds)
_DEFAULT_TTL: dict[str, int] = {
    "search_shifts":    300,   # 5 min — shifts change infrequently
    "get_schedule":     120,   # 2 min
    "analyze_coverage": 600,   # 10 min — expensive LLM call
    "lcel_chain":       900,   # 15 min
    "default":          180,
}


def _make_key(namespace: str, payload: Any) -> str:
    """Deterministic cache key: namespace + SHA-256 of canonical JSON."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"mcp:{namespace}:{digest}"


# ── In-memory fallback ────────────────────────────────────────────────────────


class InMemoryCache:
    """Simple TTL dict — used when Redis is unavailable."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}

    async def get(self, namespace: str, payload: Any) -> Any | None:
        key = _make_key(namespace, payload)
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        log.debug("cache_hit", backend="memory", namespace=namespace)
        return value

    async def set(self, namespace: str, payload: Any, value: Any, ttl: int | None = None) -> None:
        key = _make_key(namespace, payload)
        ttl = ttl or _DEFAULT_TTL.get(namespace, _DEFAULT_TTL["default"])
        self._store[key] = (value, time.monotonic() + ttl)
        log.debug("cache_set", backend="memory", namespace=namespace, ttl=ttl)

    async def delete(self, namespace: str, payload: Any) -> None:
        key = _make_key(namespace, payload)
        self._store.pop(key, None)

    async def flush(self) -> None:
        self._store.clear()

    async def close(self) -> None:
        pass


# ── Redis cache ───────────────────────────────────────────────────────────────


class RedisCache:
    """Async Redis cache using redis-py (async client)."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )

    async def get(self, namespace: str, payload: Any) -> Any | None:
        key = _make_key(namespace, payload)
        try:
            raw = await self._client.get(key)
            if raw is None:
                return None
            log.debug("cache_hit", backend="redis", namespace=namespace)
            return json.loads(raw)
        except Exception as exc:
            log.warning("cache_get_error", backend="redis", error=str(exc))
            return None

    async def set(self, namespace: str, payload: Any, value: Any, ttl: int | None = None) -> None:
        key = _make_key(namespace, payload)
        ttl = ttl or _DEFAULT_TTL.get(namespace, _DEFAULT_TTL["default"])
        try:
            await self._client.setex(key, ttl, json.dumps(value, default=str))
            log.debug("cache_set", backend="redis", namespace=namespace, ttl=ttl)
        except Exception as exc:
            log.warning("cache_set_error", backend="redis", error=str(exc))

    async def delete(self, namespace: str, payload: Any) -> None:
        key = _make_key(namespace, payload)
        try:
            await self._client.delete(key)
        except Exception as exc:
            log.warning("cache_delete_error", backend="redis", error=str(exc))

    async def flush(self) -> None:
        try:
            await self._client.flushdb()
        except Exception:
            pass

    async def close(self) -> None:
        await self._client.aclose()


# ── Factory ───────────────────────────────────────────────────────────────────

_cache_instance: RedisCache | InMemoryCache | None = None


async def get_cache() -> RedisCache | InMemoryCache:
    """Return a connected cache instance (Redis if available, else in-memory)."""
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    settings = get_settings()
    if settings.redis_url and settings.redis_url != "redis://localhost:6379":
        try:
            candidate = RedisCache(settings.redis_url)
            import redis.asyncio as aioredis
            await candidate._client.ping()
            log.info("cache_backend", backend="redis", url=settings.redis_url)
            _cache_instance = candidate
            return _cache_instance
        except Exception as exc:
            log.warning("redis_unavailable", error=str(exc), fallback="in-memory")

    log.info("cache_backend", backend="in-memory")
    _cache_instance = InMemoryCache()
    return _cache_instance


def reset_cache() -> None:
    """Reset singleton — used in tests."""
    global _cache_instance
    _cache_instance = None
