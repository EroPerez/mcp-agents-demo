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
from src.agents.langchain_agent import LangChainDemos
from src.agents.router_agent import RouterAgent
from src.agents.crewai_agent import run_crew_demo
from src.agents.autogen_agent import run_autogen_demo
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


async def demo_langchain() -> None:
    """Demo 5: LangChain — LCEL chain, tool-calling agent, streaming, parallel chains."""
    console.rule("[bold cyan]Demo 5 — LangChain (LCEL · Tool Agent · Streaming · Parallel)[/]")
    from src.agents.langchain_agent import LangChainDemos

    results = await LangChainDemos().run_all()

    # 5a LCEL chain
    console.print(Panel(
        results["lcel_chain"],
        title="5a · LCEL Chain — shift supervisor briefing",
        border_style="magenta",
    ))

    # 5b Tool-calling agent
    agent = results["tool_agent"]
    console.print(Panel(
        f"[bold]Query:[/] {agent['query']}\n\n"
        f"[bold]Answer:[/] {agent['answer']}\n\n"
        f"[dim]Tool calls made: {agent['intermediate_steps']}[/]",
        title="5b · Tool-Calling Agent",
        border_style="magenta",
    ))

    # 5c Streaming
    console.print(Panel(
        results["streaming"],
        title="5c · Streaming (astream_events)",
        border_style="magenta",
    ))

    # 5d Parallel chains
    t = Table(title="5d · Parallel LCEL Chains (3 agencies concurrently)")
    t.add_column("Agency ID")
    t.add_column("Summary")
    for row in results["parallel_chains"]:
        t.add_row(str(row.get("agency_id", "")), row.get("summary", ""))
    console.print(t)


async def demo_router_agent() -> None:
    """Demo 6: Multi-agent handoff — Router → Specialist."""
    console.rule("[bold cyan]Demo 6 — Multi-Agent Handoff (Router → Specialist)[/]")
    router = RouterAgent()

    queries = [
        ("What open shifts are available for agency 42 next week?",  42),
        ("What is the coverage risk for agency 10 in January 2025?", 10),
        ("Generate an executive report for agencies 1, 2, and 3.",   1),
        ("What is the meaning of life?",                             1),
    ]

    t = Table(title="Router → Specialist Handoff Results")
    t.add_column("Query", max_width=40)
    t.add_column("Domain", style="bold")
    t.add_column("Answer", max_width=60)

    results = await router.run_batch(queries, max_concurrent=2)
    for (query, _), result in zip(queries, results):
        domain_color = {
            "scheduling": "cyan", "coverage": "yellow",
            "reporting": "green", "unknown": "red",
        }.get(result.domain, "white")
        t.add_row(
            query[:40] + ("…" if len(query) > 40 else ""),
            f"[{domain_color}]{result.domain}[/]",
            result.answer[:60] + ("…" if len(result.answer) > 60 else ""),
        )
    console.print(t)


async def demo_cache() -> None:
    """Demo 7: Redis-backed semantic cache (in-memory fallback)."""
    console.rule("[bold cyan]Demo 7 — Semantic Cache (Redis / in-memory fallback)[/]")
    from src.core.cache import get_cache

    cache = await get_cache()
    payload = {"agency_id": 42, "date_from": "2025-01-01", "date_to": "2025-01-07"}

    # First call — cache miss
    hit = await cache.get("analyze_coverage", payload)
    console.print(f"  Cache miss: {hit is None}")

    # Populate cache
    from src.tools.shift_tools import analyze_coverage
    result = await analyze_coverage(**payload)
    await cache.set("analyze_coverage", payload, result)

    # Second call — cache hit
    hit = await cache.get("analyze_coverage", payload)
    console.print(f"  Cache hit:  {hit is not None}")
    console.print(Panel(
        f"coverage_pct={hit['coverage_pct']}%  risk={hit['risk_level']}",
        title="Cached result",
        border_style="green",
    ))

    await cache.flush()


async def demo_tracing() -> None:
    """Demo 8: OpenTelemetry tracing with @traced decorator."""
    console.rule("[bold cyan]Demo 8 — OpenTelemetry Tracing (console exporter)[/]")
    from src.core.tracing import configure_tracing, get_tracer, traced

    configure_tracing()
    tracer = get_tracer(__name__)

    @traced("demo.analyze_coverage", attributes={"service": "mcp-agents-demo"})
    async def traced_analysis(agency_id: int) -> dict:
        from src.tools.shift_tools import analyze_coverage
        return await analyze_coverage(agency_id, "2025-01-01", "2025-01-07")

    with tracer.start_as_current_span("demo.runner") as root:
        root.set_attribute("demo.name", "tracing")
        result = await traced_analysis(42)

    console.print(Panel(
        f"Span 'demo.runner' → 'demo.analyze_coverage' exported.\n"
        f"coverage_pct={result['coverage_pct']}% | risk={result['risk_level']}\n\n"
        f"[dim]In production: set APP_ENV=production and add OTEL_EXPORTER_OTLP_ENDPOINT\n"
        f"to export to Jaeger → http://localhost:16686[/]",
        title="OTel Trace",
        border_style="blue",
    ))


async def demo_portkey() -> None:
    """Demo 9: Portkey gateway — fallback routing + semantic cache + guardrails."""
    console.rule("[bold cyan]Demo 9 — Portkey Gateway (fallback · cache · guardrails)[/]")
    from src.gateway.portkey_client import get_portkey_client

    client = get_portkey_client()
    messages = [
        {"role": "system", "content": "You are a scheduling assistant."},
        {"role": "user",   "content": "Summarize coverage risks for agency 42 in January 2025."},
    ]
    result = await client.complete(messages, agency_id=42, feature="reporting")

    console.print(Panel(
        result.data,
        title=f"Portkey response | model={result.model_used} | {result.duration_ms:.0f}ms",
        border_style="magenta",
    ))


async def demo_crewai() -> None:
    """Demo 10: CrewAI — 3-agent sequential crew (Analyst → Strategist → Reporter)."""
    console.rule("[bold cyan]Demo 10 — CrewAI Multi-Agent Crew[/]")

    result = await run_crew_demo(agency_id=42, date_from="2025-01-01", date_to="2025-01-31")

    t = Table(title=f"Crew result — agency {result['agency_id']} | mode: {result['mode']}")
    t.add_column("Agent", style="bold cyan")
    t.add_column("Role")
    for agent in result.get("agents_used", []):
        role_map = {
            "Coverage Analyst": "Fetches & analyzes coverage data",
            "Risk Strategist":   "Builds prioritized action plan",
            "Executive Report Writer": "Writes leadership briefing",
        }
        t.add_row(agent, role_map.get(agent, ""))
    console.print(t)

    console.print(Panel(
        result.get("executive_briefing", ""),
        title="Executive Briefing (Reporter Agent output)",
        border_style="green",
    ))


async def demo_autogen() -> None:
    """Demo 11: AutoGen — RoundRobinGroupChat with tool-calling agents."""
    console.rule("[bold cyan]Demo 11 — AutoGen RoundRobinGroupChat[/]")

    result = await run_autogen_demo(agency_id=42, date_from="2025-01-01", date_to="2025-01-31")

    console.print(f"  Team: [bold]{result['team']}[/]  |  Messages exchanged: [bold]{result['messages']}[/]  |  Mode: {result['mode']}")

    for turn in result.get("conversation", []):
        agent_color = {"DataAgent": "cyan", "AnalystAgent": "yellow", "PlannerAgent": "green"}.get(turn["agent"], "white")
        console.print(Panel(
            turn["message"],
            title=f"[{agent_color}]{turn['agent']}[/]",
            border_style=agent_color,
            padding=(0, 1),
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
        await demo_langchain()
        await demo_router_agent()
        await demo_cache()
        await demo_tracing()
        await demo_portkey()
        await demo_crewai()
        await demo_autogen()
    finally:
        shutdown_executors(wait=False)

    console.print("\n[bold green]✓ All demos completed successfully.[/]")


def run_demo() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run_demo()
