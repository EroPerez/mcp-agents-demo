"""FastMCP server — exposes tools, resources and prompt templates.

Transport options:
  stdio  → Claude Desktop / MCP Inspector
  sse    → HTTP production deployment

Run:
    uv run mcp-server          # uses MCP_TRANSPORT env var (default: stdio)
    uv run python -m src.server.main --transport sse --port 8000
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastmcp import Context, FastMCP

from src.agents.coverage_agent import CoverageAgent
from src.core.config import get_settings
from src.core.logging import configure_logging
from src.core.models import ShiftQuery
from src.tools.shift_tools import (
    analyze_coverage,
    get_schedule,
    search_shifts,
    update_shift_status,
)

log = structlog.get_logger(__name__)
settings = get_settings()

# ── Server instantiation ──────────────────────────────────────────────────────

mcp = FastMCP(
    name="mcp-agents-demo",
    version="0.1.0",
    instructions=(
        "Scheduling assistant for emergency services agencies. "
        "Use search_shifts to find available shifts, get_schedule for daily views, "
        "update_shift_status for mutations, and analyze_coverage for insights."
    ),
)


# ── Tools ─────────────────────────────────────────────────────────────────────


@mcp.tool()
async def tool_search_shifts(
    agency_id: int,
    date_from: str,
    date_to: str,
    positions: list[str] | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Search open shifts for an agency within a date range.

    Args:
        agency_id: Agency identifier (positive integer).
        date_from: Start date in YYYY-MM-DD format.
        date_to:   End date in YYYY-MM-DD format.
        positions: Optional list of position codes to filter by.
    """
    if ctx:
        await ctx.info(f"Searching shifts for agency {agency_id}")
    query = ShiftQuery(
        agency_id=agency_id,
        date_from=date_from,
        date_to=date_to,
        positions=positions or [],
    )
    return await search_shifts(query)


@mcp.tool()
async def tool_get_schedule(
    agency_id: int,
    date: str,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Get the complete daily schedule for an agency.

    Args:
        agency_id: Agency identifier.
        date: Target date in YYYY-MM-DD format.
    """
    if ctx:
        await ctx.info(f"Fetching schedule: agency={agency_id}, date={date}")
    return await get_schedule(agency_id, date)


@mcp.tool()
async def tool_update_shift_status(
    shift_id: int,
    status: str,
    notes: str = "",
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Update a shift's status.

    Args:
        shift_id: Unique shift identifier.
        status:   New status — 'open', 'filled', or 'cancelled'.
        notes:    Optional operator notes.
    """
    if ctx:
        await ctx.info(f"Updating shift {shift_id} -> {status}")
    return await update_shift_status(shift_id, status, notes)


@mcp.tool()
async def tool_analyze_coverage(
    agency_id: int,
    date_from: str,
    date_to: str,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Analyze staffing coverage and identify gaps.

    Returns a structured CoverageAnalysis with risk level and recommendations.

    Args:
        agency_id: Agency identifier.
        date_from: Start date YYYY-MM-DD.
        date_to:   End date YYYY-MM-DD.
    """
    if ctx:
        await ctx.info("Running coverage analysis…")
    return await analyze_coverage(agency_id, date_from, date_to)


@mcp.tool()
async def tool_ai_analyze_coverage(
    agency_id: int,
    date_from: str,
    date_to: str,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Run the AI Coverage Agent — uses LLM to provide deeper insights.

    Uses pydantic-ai internally; falls back to direct analysis in demo mode.

    Args:
        agency_id: Agency identifier.
        date_from: Start date YYYY-MM-DD.
        date_to:   End date YYYY-MM-DD.
    """
    if ctx:
        await ctx.info("Invoking AI Coverage Agent…")
    agent = CoverageAgent()
    result = await agent.run(agency_id, date_from, date_to)
    return {
        "analysis": result.data.model_dump(),
        "model_used": result.model_used,
        "cost_usd": result.cost_usd,
        "duration_ms": result.duration_ms,
    }


# ── Resources ─────────────────────────────────────────────────────────────────


@mcp.resource("schedule://{agency_id}/{date}")
async def resource_schedule(agency_id: int, date: str) -> str:
    """Expose a daily schedule as a readable resource.

    URI pattern: schedule://{agency_id}/{date}
    Example:     schedule://42/2025-01-15
    """
    data = await get_schedule(agency_id, date)
    return json.dumps(data, indent=2)


# ── Prompt templates ──────────────────────────────────────────────────────────


@mcp.prompt()
def prompt_analyze_coverage(agency_id: int, date_from: str, date_to: str) -> str:
    """Prompt template for coverage analysis requests."""
    return (
        f"Analyze shift coverage for agency {agency_id} from {date_from} to {date_to}. "
        "Use the analyze_coverage tool to get data, then:\n"
        "1. Summarize the overall coverage percentage and risk level\n"
        "2. List the most critical gaps by position\n"
        "3. Provide 3 specific, actionable recommendations\n"
        "Be concise and data-driven."
    )


@mcp.prompt()
def prompt_daily_briefing(agency_id: int, date: str) -> str:
    """Prompt template for daily scheduling briefings."""
    return (
        f"Generate a daily scheduling briefing for agency {agency_id} on {date}.\n"
        "Use get_schedule to fetch the data, then provide:\n"
        "- Coverage summary by position\n"
        "- Any open (unfilled) shifts\n"
        "- Recommended actions for the shift supervisor"
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def run_server() -> None:
    configure_logging()
    log.info(
        "mcp_server_starting",
        transport=settings.mcp_transport,
        demo_mode=settings.demo_mode,
    )
    mcp.run(transport=settings.mcp_transport)


if __name__ == "__main__":
    run_server()
