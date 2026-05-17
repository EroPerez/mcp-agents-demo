"""Multi-agent handoff — Router → Specialist pattern.

Architecture:
    User query
        │
        ▼
    RouterAgent          ← lightweight LLM call, classifies intent
        │
        ├─► SchedulingAgent   ← shift search, schedule view
        ├─► CoverageAgent     ← coverage analysis, risk assessment
        ├─► ReportingAgent    ← summaries, executive briefings
        └─► (unknown)         ← polite fallback

Each specialist is independent and testable in isolation.
The router uses structured output (Pydantic) to guarantee a valid domain.

Demonstrates:
- Multi-agent orchestration without a framework
- Generic Agent[InputT, OutputT] pattern
- asyncio concurrency: router + specialist run with timeout
- Conversation history (multi-turn context passing)
- contextvars for request_id propagation across agents
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

from src.core.concurrency import gather_with_limit
from src.core.config import get_settings
from src.core.models import AgentResult, CoverageAnalysis
from src.tools.shift_tools import analyze_coverage, get_schedule, search_shifts

log = structlog.get_logger(__name__)


# ── Intent classification ─────────────────────────────────────────────────────

Domain = Literal["scheduling", "coverage", "reporting", "unknown"]


class RouterDecision(BaseModel):
    domain: Domain
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ── Base specialist protocol ──────────────────────────────────────────────────


@dataclass
class AgentContext:
    request_id: str
    agency_id: int
    history: list[dict[str, str]] = field(default_factory=list)


class SpecialistResult(BaseModel):
    domain: Domain
    answer: str
    data: dict[str, Any] = Field(default_factory=dict)
    tools_called: list[str] = Field(default_factory=list)
    model_used: str = "demo"


# ── Specialists ───────────────────────────────────────────────────────────────


class SchedulingSpecialist:
    """Handles shift search and schedule queries."""

    async def run(self, query: str, ctx: AgentContext) -> SpecialistResult:
        log.info("specialist_run", domain="scheduling", agency=ctx.agency_id)

        # In real mode: LLM extracts params; in demo: use defaults
        schedule = await get_schedule(ctx.agency_id, "2025-01-15")
        from src.core.models import ShiftQuery
        shifts = await search_shifts(
            ShiftQuery(agency_id=ctx.agency_id, date_from="2025-01-01", date_to="2025-01-07")
        )

        answer = (
            f"Agency {ctx.agency_id} has {schedule['total']} scheduled positions on 2025-01-15. "
            f"There are {len(shifts)} open shifts available in the first week of January."
        )
        return SpecialistResult(
            domain="scheduling",
            answer=answer,
            data={"schedule": schedule, "open_shifts_count": len(shifts)},
            tools_called=["get_schedule", "search_shifts"],
        )


class CoverageSpecialist:
    """Handles coverage analysis and risk assessment."""

    async def run(self, query: str, ctx: AgentContext) -> SpecialistResult:
        log.info("specialist_run", domain="coverage", agency=ctx.agency_id)

        raw = await analyze_coverage(ctx.agency_id, "2025-01-01", "2025-01-31")
        analysis = CoverageAnalysis(**raw)

        answer = (
            f"Agency {ctx.agency_id} January coverage: {analysis.coverage_pct}% "
            f"({analysis.risk_level} risk). "
            f"Critical gaps: {', '.join(analysis.critical_gaps[:3]) or 'none'}. "
            f"Recommendation: {analysis.recommendations[0]}."
        )
        return SpecialistResult(
            domain="coverage",
            answer=answer,
            data=raw,
            tools_called=["analyze_coverage"],
        )


class ReportingSpecialist:
    """Handles executive summaries and multi-agency reports."""

    async def run(self, query: str, ctx: AgentContext) -> SpecialistResult:
        log.info("specialist_run", domain="reporting", agency=ctx.agency_id)

        # Run coverage for multiple agencies concurrently
        agency_ids = [ctx.agency_id, ctx.agency_id + 1, ctx.agency_id + 2]
        coros = [analyze_coverage(aid, "2025-01-01", "2025-01-31") for aid in agency_ids]
        results = await gather_with_limit(coros, max_concurrent=3)

        avg_coverage = sum(r["coverage_pct"] for r in results) / len(results)
        risk_counts: dict[str, int] = {}
        for r in results:
            risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1

        answer = (
            f"Executive report — {len(agency_ids)} agencies, January 2025:\n"
            f"• Average coverage: {avg_coverage:.1f}%\n"
            f"• Risk distribution: {risk_counts}\n"
            f"• {sum(len(r['critical_gaps']) for r in results)} total gaps identified"
        )
        return SpecialistResult(
            domain="reporting",
            answer=answer,
            data={"agencies": agency_ids, "avg_coverage": avg_coverage, "risk_counts": risk_counts},
            tools_called=["analyze_coverage (×3 parallel)"],
        )


# ── Router ─────────────────────────────────────────────────────────────────────


_KEYWORD_MAP: dict[str, Domain] = {
    "shift":      "scheduling",
    "schedule":   "scheduling",
    "open":       "scheduling",
    "available":  "scheduling",
    "coverage":   "coverage",
    "risk":       "coverage",
    "gap":        "coverage",
    "fill":       "coverage",
    "report":     "reporting",
    "summary":    "reporting",
    "executive":  "reporting",
    "overview":   "reporting",
}


def _classify_local(query: str) -> RouterDecision:
    """Keyword-based classifier — used in demo mode (no LLM call)."""
    q = query.lower()
    scores: dict[Domain, int] = {"scheduling": 0, "coverage": 0, "reporting": 0, "unknown": 0}
    for keyword, domain in _KEYWORD_MAP.items():
        if keyword in q:
            scores[domain] += 1

    best: Domain = max(scores, key=lambda d: scores[d])  # type: ignore[arg-type]
    if scores[best] == 0:
        best = "unknown"

    total = sum(scores.values()) or 1
    return RouterDecision(
        domain=best,
        confidence=round(scores[best] / total, 2),
        reasoning=f"keyword match: {scores}",
    )


async def _classify_llm(query: str) -> RouterDecision:
    """LLM-based classifier for production use."""
    from src.gateway.llm_client import get_llm_client

    client = get_llm_client()
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the user query into exactly one domain. "
                "Reply with JSON only: {\"domain\": \"scheduling|coverage|reporting|unknown\", "
                "\"confidence\": 0.0-1.0, \"reasoning\": \"...\"}"
            ),
        },
        {"role": "user", "content": query},
    ]
    result = await client.complete(messages, max_tokens=128, temperature=0.0)
    try:
        return RouterDecision.model_validate_json(result.data)
    except Exception:
        return _classify_local(query)  # fallback


# ── RouterAgent orchestrator ───────────────────────────────────────────────────


class RouterAgent:
    """Top-level orchestrator that routes queries to the right specialist.

    Handoff flow:
        1. Classify query (LLM in prod, keyword in demo)
        2. Select specialist
        3. Run specialist with timeout
        4. Return typed SpecialistResult
    """

    _SPECIALISTS: dict[Domain, type] = {
        "scheduling": SchedulingSpecialist,
        "coverage":   CoverageSpecialist,
        "reporting":  ReportingSpecialist,
    }

    def __init__(self) -> None:
        self._settings = get_settings()

    async def run(self, query: str, agency_id: int = 1) -> SpecialistResult:
        import uuid

        ctx = AgentContext(request_id=str(uuid.uuid4()), agency_id=agency_id)

        # Step 1: classify
        if self._settings.demo_mode:
            decision = _classify_local(query)
        else:
            decision = await _classify_llm(query)

        log.info(
            "router_decision",
            domain=decision.domain,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
        )

        # Step 2: select specialist
        specialist_cls = self._SPECIALISTS.get(decision.domain)
        if specialist_cls is None:
            return SpecialistResult(
                domain="unknown",
                answer="I don't have a specialist for that query. Try asking about shifts, coverage, or reports.",
            )

        # Step 3: run with timeout
        specialist = specialist_cls()
        async with asyncio.timeout(self._settings.agent_timeout_seconds):
            result = await specialist.run(query, ctx)

        log.info(
            "handoff_complete",
            domain=result.domain,
            tools_called=result.tools_called,
        )
        return result

    async def run_batch(
        self,
        queries: list[tuple[str, int]],
        max_concurrent: int = 3,
    ) -> list[SpecialistResult]:
        """Process multiple (query, agency_id) pairs concurrently."""
        coros = [self.run(q, aid) for q, aid in queries]
        return await gather_with_limit(coros, max_concurrent=max_concurrent)
