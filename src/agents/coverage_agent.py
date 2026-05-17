"""Coverage Analysis Agent using Pydantic AI.

Demonstrates:
- pydantic-ai Agent with structured output (CoverageAnalysis)
- Tool registration via @agent.tool
- Generic AgentResult[T] wrapper
- asyncio.timeout integration
- Mock fallback for demo mode
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from src.core.config import get_settings
from src.core.models import AgentResult, CoverageAnalysis
from src.tools.shift_tools import analyze_coverage, get_schedule, search_shifts

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You are an expert in emergency services staffing and shift scheduling.
Your role is to analyze shift coverage data for fire and EMS agencies.

When analyzing coverage:
- Identify staffing gaps by position and time of day
- Assess risk levels: critical (<60%), high (<75%), medium (<90%), low (>=90%)
- Provide actionable, specific recommendations
- Always respond with valid JSON matching the CoverageAnalysis schema

Be concise and data-driven. Do not hallucinate shift data — use the provided tools.
"""


class CoverageAgent:
    """Agent that analyzes shift coverage and returns structured CoverageAnalysis."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def run(
        self,
        agency_id: int,
        date_from: str,
        date_to: str,
    ) -> AgentResult[CoverageAnalysis]:
        """Run full coverage analysis for a given agency and date range.

        Falls back to a direct tool call in demo mode (no API key needed).
        """
        log.info("coverage_agent_start", agency_id=agency_id, date_from=date_from, date_to=date_to)

        if self._settings.demo_mode:
            return await self._demo_run(agency_id, date_from, date_to)

        return await self._pydantic_ai_run(agency_id, date_from, date_to)

    async def _demo_run(
        self, agency_id: int, date_from: str, date_to: str
    ) -> AgentResult[CoverageAnalysis]:
        """Direct tool execution — no LLM call required."""
        raw = await analyze_coverage(agency_id, date_from, date_to)
        analysis = CoverageAnalysis(**raw)
        return AgentResult(
            data=analysis,
            model_used="demo (no LLM)",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            duration_ms=0.0,
        )

    async def _pydantic_ai_run(
        self, agency_id: int, date_from: str, date_to: str
    ) -> AgentResult[CoverageAnalysis]:
        """Real agent run using pydantic-ai."""
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.anthropic import AnthropicModel
        except ImportError as exc:
            raise RuntimeError("pydantic-ai not installed: pip install pydantic-ai") from exc

        import time

        model = AnthropicModel(
            "claude-haiku-4-5",
            api_key=self._settings.anthropic_api_key,
        )

        agent: Agent[None, CoverageAnalysis] = Agent(
            model,
            result_type=CoverageAnalysis,
            system_prompt=SYSTEM_PROMPT,
        )

        # Register tools on the pydantic-ai agent
        @agent.tool_plain
        async def tool_analyze_coverage(a_id: int, df: str, dt: str) -> str:
            result = await analyze_coverage(a_id, df, dt)
            return json.dumps(result)

        @agent.tool_plain
        async def tool_search_shifts(a_id: int, df: str, dt: str) -> str:
            from src.core.models import ShiftQuery

            q = ShiftQuery(agency_id=a_id, date_from=df, date_to=dt)
            result = await search_shifts(q)
            return json.dumps(result[:20])  # cap to 20 items for context

        prompt = (
            f"Analyze shift coverage for agency {agency_id} "
            f"from {date_from} to {date_to}. "
            "Use the tools to fetch real data, then return a CoverageAnalysis."
        )

        start = time.perf_counter()
        async with asyncio.timeout(self._settings.agent_timeout_seconds):
            result = await agent.run(prompt)

        duration_ms = (time.perf_counter() - start) * 1000
        usage = result.usage()

        log.info(
            "coverage_agent_done",
            agency_id=agency_id,
            risk=result.data.risk_level,
            coverage_pct=result.data.coverage_pct,
            duration_ms=round(duration_ms, 1),
        )

        return AgentResult(
            data=result.data,
            model_used="claude-haiku-4-5",
            input_tokens=getattr(usage, "request_tokens", 0),
            output_tokens=getattr(usage, "response_tokens", 0),
            cost_usd=0.0,
            duration_ms=duration_ms,
        )


# ── Parallel multi-agency analysis ────────────────────────────────────────────


async def analyze_multiple_agencies(
    agency_ids: list[int],
    date_from: str,
    date_to: str,
    max_concurrent: int = 3,
) -> list[AgentResult[CoverageAnalysis]]:
    """Analyze multiple agencies concurrently with a concurrency cap.

    Demonstrates asyncio.Semaphore + gather pattern.
    """
    from src.core.concurrency import gather_with_limit

    agent = CoverageAgent()
    coros = [agent.run(aid, date_from, date_to) for aid in agency_ids]
    return await gather_with_limit(coros, max_concurrent=max_concurrent)
