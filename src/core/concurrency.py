"""Concurrency utilities.

Demonstrates all four concurrency patterns:
- asyncio.Semaphore  → cap concurrent LLM calls
- asyncio.Queue     → producer/consumer pipeline
- ThreadPoolExecutor → off-load blocking I/O
- ProcessPoolExecutor → CPU-bound work (no GIL)
- contextvars        → propagate request context across tasks/threads
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import multiprocessing
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

import structlog

log = structlog.get_logger(__name__)

T = TypeVar("T")

# ── Shared executor singletons (created once, reused) ─────────────────────────

_thread_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="mcp-io")
_process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())

# ── contextvars ────────────────────────────────────────────────────────────────

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
agency_id_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "agency_id", default=0
)


def bind_context(request_id: str, agency_id: int = 0) -> contextvars.Token[str]:
    """Set per-task context variables. Returns token for reset."""
    agency_id_var.set(agency_id)
    return request_id_var.set(request_id)


# ── 1. Semaphore — limit concurrent LLM API calls ─────────────────────────────


class LLMSemaphore:
    """Async semaphore that caps concurrent calls to the LLM provider.

    Usage:
        sem = LLMSemaphore(max_concurrent=5)

        @sem.guard
        async def call_llm(prompt: str) -> str: ...
    """

    def __init__(self, max_concurrent: int = 5) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)

    def guard(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with self._sem:
                return await func(*args, **kwargs)

        return wrapper

    async def run(self, coro: Awaitable[T]) -> T:
        async with self._sem:
            return await coro


async def gather_with_limit(
    coros: list[Awaitable[T]],
    max_concurrent: int = 5,
) -> list[T]:
    """Run coroutines concurrently with a concurrency cap.

    Returns results in the same order as input coroutines.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def bounded(coro: Awaitable[T]) -> T:
        async with sem:
            return await coro

    return list(await asyncio.gather(*[bounded(c) for c in coros]))


# ── 2. Queue — producer / consumer pipeline ───────────────────────────────────


async def run_pipeline(
    items: list[Any],
    process_fn: Callable[[Any], Awaitable[Any]],
    n_workers: int = 4,
    queue_size: int = 0,  # 0 = unlimited
) -> list[Any]:
    """Fan-out pipeline: N workers consume from a shared queue.

    Args:
        items:      Items to process.
        process_fn: Async function applied to each item.
        n_workers:  Degree of parallelism.
        queue_size: Max queue depth (0 = unbounded).

    Returns:
        Results in completion order (may differ from input order).
    """
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_size)
    results: list[Any] = []
    lock = asyncio.Lock()

    async def producer() -> None:
        for item in items:
            await queue.put(item)
        for _ in range(n_workers):
            await queue.put(None)  # one sentinel per worker

    async def worker(worker_id: int) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            try:
                result = await process_fn(item)
                async with lock:
                    results.append(result)
            except Exception as exc:
                log.warning("pipeline_worker_error", worker=worker_id, error=str(exc))
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(n_workers)]
    await producer()
    await asyncio.gather(*workers)
    return results


# ── 3. ThreadPoolExecutor — blocking I/O in async context ────────────────────


async def run_in_thread(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a blocking (sync) function in the shared thread pool.

    Propagates the current contextvars to the thread.

    Example:
        result = await run_in_thread(requests.get, url, timeout=10)
    """
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()  # snapshot — propagates request_id etc.
    wrapped = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(_thread_pool, wrapped)


def sync_batch_with_threads(
    items: list[Any],
    func: Callable[[Any], T],
    max_workers: int = 8,
    timeout: float | None = 30.0,
) -> list[T]:
    """Synchronous batch processor using ThreadPoolExecutor.as_completed.

    For use outside of an async context (e.g. CLI scripts, pytest fixtures).
    """
    results: list[T] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(func, item): item for item in items}
        for future in as_completed(futures, timeout=timeout):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                log.error("thread_batch_error", item=str(item), error=str(exc))
    return results


# ── 4. ProcessPoolExecutor — CPU-bound work ───────────────────────────────────


# NOTE: Functions submitted to ProcessPoolExecutor MUST be defined at module
# level (not lambdas or closures) because pickle is used for IPC.


def _cpu_task(data: list[float]) -> float:
    """Example CPU-bound task: sum of squares (runs in subprocess)."""
    return sum(x**2 for x in data)


async def run_cpu_bound(
    chunks: list[list[float]],
) -> list[float]:
    """Distribute CPU-intensive work across processes.

    Each chunk is processed in a separate subprocess — true parallelism,
    bypassing the GIL.
    """
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(_process_pool, _cpu_task, chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── Cleanup ───────────────────────────────────────────────────────────────────


def shutdown_executors(wait: bool = True) -> None:
    """Graceful shutdown. Call on application exit."""
    _thread_pool.shutdown(wait=wait)
    _process_pool.shutdown(wait=wait)
