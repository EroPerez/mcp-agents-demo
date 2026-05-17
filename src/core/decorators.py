"""Reusable decorators for MCP tools and agent calls.

Demonstrates:
- functools.wraps
- Decorator factories (parametrized decorators)
- Class-based decorators
- Stacking order
- Tool registry pattern
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

log = structlog.get_logger(__name__)

# ── Tool Registry ─────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Callable] = {}


def tool(name: str | None = None, description: str = "") -> Callable[[F], F]:
    """Register a coroutine as a named tool.

    Usage:
        @tool(description="Searches shifts by agency and date")
        async def search_shifts(agency_id: int, date: str) -> list[dict]: ...
    """

    def decorator(func: F) -> F:
        key = name or func.__name__
        func._tool_meta = {"name": key, "description": description or func.__doc__ or ""}  # type: ignore[attr-defined]
        _TOOL_REGISTRY[key] = func

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def get_tool(name: str) -> Callable | None:
    return _TOOL_REGISTRY.get(name)


def list_tools() -> list[dict]:
    return [getattr(fn, "_tool_meta", {"name": k}) for k, fn in _TOOL_REGISTRY.items()]


# ── Timer decorator ────────────────────────────────────────────────────────────


def timed(func: F) -> F:
    """Log execution time of an async function."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log.debug("timed", fn=func.__name__, elapsed_ms=round(elapsed * 1000, 2))
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            log.warning("timed_error", fn=func.__name__, elapsed_ms=round(elapsed * 1000, 2))
            raise

    return wrapper  # type: ignore[return-value]


# ── Rate limiter (class-based decorator) ──────────────────────────────────────


class RateLimit:
    """Token-bucket rate limiter for async callables.

    Usage:
        @RateLimit(calls_per_second=10)
        async def call_llm(prompt: str) -> str: ...
    """

    def __init__(self, calls_per_second: float) -> None:
        self.min_interval = 1.0 / calls_per_second
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self._lock:
                now = asyncio.get_event_loop().time()
                wait = self.min_interval - (now - self._last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_call = asyncio.get_event_loop().time()
            return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


# ── Retry with exponential backoff ───────────────────────────────────────────


def llm_retry(
    max_attempts: int = 4,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
) -> Callable[[F], F]:
    """Retry decorator tuned for LLM API calls (rate limits, transient errors).

    Stacking example:
        @timed                          # 3rd: wraps retry+ratelimit
        @llm_retry(max_attempts=4)      # 2nd: wraps ratelimit
        @RateLimit(calls_per_second=5)  # 1st: closest to the function
        async def call_llm(...): ...
    """
    import anthropic

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APIConnectionError, asyncio.TimeoutError)
        ),
        reraise=True,
    )


# ── Timeout wrapper ───────────────────────────────────────────────────────────


def with_timeout(seconds: float) -> Callable[[F], F]:
    """Enforce an async timeout on any coroutine function."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with asyncio.timeout(seconds):
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
