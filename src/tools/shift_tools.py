"""Scheduling domain tools.

Each function is:
  1. Registered in the tool registry via @tool
  2. Mounted on the FastMCP server in src/server/main.py
  3. Independently unit-testable

Demonstrates: decorators stacking, Pydantic validation, async patterns.
"""

from __future__ import annotations

import asyncio
import random
from datetime import date, timedelta

import structlog

from src.core.decorators import RateLimit, timed, tool
from src.core.models import CoverageAnalysis, Shift, ShiftQuery

log = structlog.get_logger(__name__)

# ── In-memory demo database ──────────────────────────────────────────────────

_POSITIONS = ["Paramedic", "EMT", "Firefighter", "Captain", "Engineer"]
_STATUSES: list = ["open", "filled", "filled", "filled", "cancelled"]  # weighted


def _generate_shifts(agency_id: int, date_from: str, date_to: str) -> list[Shift]:
    """Generate deterministic fake shifts for demo/testing."""
    shifts = []
    start = date.fromisoformat(date_from)
    end = date.fromisoformat(date_to)
    shift_id = agency_id * 1000

    current = start
    while current <= end:
        for position in _POSITIONS:
            rng = random.Random(f"{agency_id}-{current}-{position}")
            status = rng.choice(_STATUSES)
            shifts.append(
                Shift(
                    shift_id=shift_id,
                    agency_id=agency_id,
                    position=position,
                    date=str(current),
                    start_time="07:00",
                    end_time="19:00",
                    status=status,
                    assigned_to=f"staff_{rng.randint(100,999)}" if status == "filled" else None,
                )
            )
            shift_id += 1
        current += timedelta(days=1)

    return shifts


# ── Tools ─────────────────────────────────────────────────────────────────────


@timed
@tool(description="Search available shifts for an agency within a date range.")
async def search_shifts(query: ShiftQuery) -> list[dict]:
    """Search available shifts.

    Args:
        query: ShiftQuery with agency_id, date_from, date_to, positions filter.

    Returns:
        List of shift dicts with open status.
    """
    await asyncio.sleep(0.05)  # simulate DB latency

    shifts = _generate_shifts(query.agency_id, query.date_from, query.date_to)

    if query.positions:
        shifts = [s for s in shifts if s.position in query.positions]

    open_shifts = [s for s in shifts if s.status == "open"]
    log.info("search_shifts", agency=query.agency_id, found=len(open_shifts))
    return [s.model_dump() for s in open_shifts]


@timed
@tool(description="Get the full schedule for an agency on a specific date.")
async def get_schedule(agency_id: int, date: str) -> dict:
    """Return complete schedule for one agency/date.

    Args:
        agency_id: Agency identifier.
        date: Target date in YYYY-MM-DD format.
    """
    await asyncio.sleep(0.03)
    shifts = _generate_shifts(agency_id, date, date)
    by_position: dict[str, list] = {}
    for s in shifts:
        by_position.setdefault(s.position, []).append(s.model_dump())

    return {
        "agency_id": agency_id,
        "date": date,
        "total": len(shifts),
        "by_position": by_position,
    }


@timed
@RateLimit(calls_per_second=2.0)  # mutation: max 2 writes/sec
@tool(description="Update the status of a shift.")
async def update_shift_status(shift_id: int, status: str, notes: str = "") -> dict:
    """Update a shift's status.

    Args:
        shift_id: Unique shift identifier.
        status: New status — 'open', 'filled', or 'cancelled'.
        notes: Optional operator notes.
    """
    allowed_statuses = {"open", "filled", "cancelled"}
    if status not in allowed_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of {allowed_statuses}.")

    await asyncio.sleep(0.02)
    log.info("update_shift_status", shift_id=shift_id, new_status=status)
    return {"shift_id": shift_id, "status": status, "notes": notes, "updated": True}


@timed
@tool(description="Analyze staffing coverage for an agency over a date range.")
async def analyze_coverage(agency_id: int, date_from: str, date_to: str) -> dict:
    """Compute coverage statistics and identify gaps.

    Args:
        agency_id: Agency identifier.
        date_from: Start date YYYY-MM-DD.
        date_to:   End date YYYY-MM-DD.
    """
    await asyncio.sleep(0.08)
    shifts = _generate_shifts(agency_id, date_from, date_to)
    total = len(shifts)
    filled = sum(1 for s in shifts if s.status == "filled")
    pct = (filled / total * 100) if total else 0.0

    gaps = [
        f"{s.position} on {s.date}"
        for s in shifts
        if s.status == "open"
    ][:5]  # top 5

    risk: str
    if pct >= 90:
        risk = "low"
    elif pct >= 75:
        risk = "medium"
    elif pct >= 60:
        risk = "high"
    else:
        risk = "critical"

    analysis = CoverageAnalysis(
        agency_id=agency_id,
        date_range=f"{date_from}/{date_to}",
        total_shifts=total,
        filled_shifts=filled,
        coverage_pct=round(pct, 1),
        critical_gaps=gaps,
        recommendations=[
            "Increase recruitment for night shifts" if pct < 80 else "Coverage is adequate"
        ],
        risk_level=risk,
    )
    return analysis.model_dump()
