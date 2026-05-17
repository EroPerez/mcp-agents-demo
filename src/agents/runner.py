"""Demo runner — orchestrates all agents and shows concurrency patterns.

Run with:
    uv run python -m src.agents.runner
or:
    uv run demo-agents
"""

from __future__ import annotations

import asyncio

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agents.coverage_agent import CoverageAgent, analyze_multiple_agencies
from src.core.concurrency import run_pipeline, shutdown_executors
from src.core.config import get_settings
from src.core.logging import configure_logging
from src.core.models import ShiftQuery
from src.tools.shift_tools import search_shifts

console = Console()
log = structlog.get_logger(__name__)


async def demo_single_agent() -> None:
    """Demo 1: single structured-output agent."""
    console.rule("[bold cyan]Demo 1 — Coverage Agent (single agency)[/]")
    agent = CoverageAgent()
    result = await agent.run(agency_id=42, date_from="2025-01-01", date_to="2025-01-07")

    t = Table(title=f"Coverage Report — Agency 42 | Model: {result.model_used}")
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    analysis = result.data
    t.add_row("Coverage", f"{analysis.coverage_pct}%")
    t.add_row("Risk Level", f"[{'red' if analysis.risk_level in ('high','critical') else 'green'}]{analysis.risk_level}[/]")
    t.add_row("Total Shifts", str(analysis.total_shifts))
    t.add_row("Filled Shifts", str(analysis.filled_shifts))
    t.add_row("Critical Gaps", "\n".join(analysis.critical_gaps) or "None")
    t.add_row("Recommendations", "\n".join(analysis.recommendations))
    t.add_row("Cost (USD)", f"${result.cost_usd:.6f}")
    t.add_row("Duration (ms)", f"{result.duration_ms:.1f}")
    console.print(t)


async def demo_parallel_agents() -> None:
    """Demo 2: analyze 5 agencies concurrently (Semaphore-limited)."""
    console.rule("[bold cyan]Demo 2 — Parallel Multi-Agency Analysis[/]")
    agency_ids = [10, 20, 30, 40, 50]
    results = await analyze_multiple_agencies(
        agency_ids,
        date_from="2025-01-01",
        date_to="2025-01-31",
        max_concurrent=3,
    )

    t = Table(title="Multi-Agency Coverage Summary")
    t.add_column("Agency ID")
    t.add_column("Coverage %")
    t.add_column("Risk")
    t.add_column("Gaps")
    for r in results:
        a = r.data
        risk_color = {"low": "green", "medium": "yellow", "high": "orange3", "critical": "red"}.get(a.risk_level, "white")
        t.add_row(
            str(a.agency_id),
            f"{a.coverage_pct}%",
            f"[{risk_color}]{a.risk_level}[/]",
            str(len(a.critical_gaps)),
        )
    console.print(t)


async def demo_pipeline() -> None:
    """Demo 3: Queue-based pipeline processing multiple shift queries."""
    console.rule("[bold cyan]Demo 3 — Pipeline (Queue + N workers)[/]")

    queries = [
        ShiftQuery(agency_id=i, date_from="2025-01-01", date_to="2025-01-03")
        for i in range(1, 9)
    ]

    async def process_query(q: ShiftQuery) -> dict:
        shifts = await search_shifts(q)
        return {"agency_id": q.agency_id, "open_shifts": len(shifts)}

    results = await run_pipeline(queries, process_query, n_workers=4)

    t = Table(title="Pipeline Results (4 concurrent workers)")
    t.add_column("Agency ID")
    t.add_column("Open Shifts")
    for r in sorted(results, key=lambda x: x["agency_id"]):
        t.add_row(str(r["agency_id"]), str(r["open_shifts"]))
    console.print(t)


async def demo_thread_executor() -> None:
    """Demo 4: blocking I/O offloaded to ThreadPoolExecutor."""
    console.rule("[bold cyan]Demo 4 — ThreadPoolExecutor (blocking I/O)[/]")
    from src.core.concurrency import run_in_thread

    def blocking_io_task(item: int) -> str:
        import time
        time.sleep(0.05)  # simulate blocking DB call
        return f"processed_{item}"

    tasks = [run_in_thread(blocking_io_task, i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    console.print(Panel(
        "\n".join(results),
        title="ThreadPoolExecutor results (10 items, run concurrently)",
        border_style="blue",
    ))


async def main() -> None:
    configure_logging()
    settings = get_settings()

    console.print(Panel(
        f"[bold]mcp-agents-demo[/]\n"
        f"env={settings.app_env} | demo_mode={settings.demo_mode} | "
        f"max_concurrent={settings.max_concurrent_tools}",
        border_style="cyan",
    ))

    try:
        await demo_single_agent()
        await demo_parallel_agents()
        await demo_pipeline()
        await demo_thread_executor()
    finally:
        shutdown_executors(wait=False)

    console.print("\n[bold green]✓ All demos completed successfully.[/]")


def run_demo() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run_demo()
