"""CrewAI Multi-Agent Crew Demo.

Demonstrates:
- Crew with multiple Agents (Analyst, Strategist, Reporter)
- Tasks with expected_output and context dependencies
- Sequential and hierarchical Process types
- Custom @tool functions for CrewAI agents
- Mock execution in demo mode (no API key needed)

Architecture:
    ┌─────────────────────────────────────────┐
    │  CoverageAnalystAgent                   │
    │  → collects shift + coverage data       │
    └──────────────────┬──────────────────────┘
                       │ context
    ┌──────────────────▼──────────────────────┐
    │  RiskStrategistAgent                    │
    │  → evaluates risks, builds action plan  │
    └──────────────────┬──────────────────────┘
                       │ context
    ┌──────────────────▼──────────────────────┐
    │  ReportWriterAgent                      │
    │  → produces executive briefing          │
    └─────────────────────────────────────────┘
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from src.core.config import get_settings

log = structlog.get_logger(__name__)


# ── CrewAI tools (plain functions decorated with @tool) ───────────────────────

def _build_crewai_tools() -> list:
    """Build CrewAI-compatible tools lazily."""
    from crewai.tools import tool as crewai_tool

    @crewai_tool("AnalyzeCoverage")
    def analyze_coverage_tool(agency_id: int, date_from: str, date_to: str) -> str:
        """Analyze staffing coverage for an agency over a date range.
        Returns coverage percentage, risk level, and critical gaps.
        """
        import asyncio
        from src.tools.shift_tools import analyze_coverage
        result = asyncio.run(analyze_coverage(agency_id, date_from, date_to))
        return json.dumps(result, indent=2)

    @crewai_tool("SearchShifts")
    def search_shifts_tool(agency_id: int, date_from: str, date_to: str) -> str:
        """Search open shifts for an agency within a date range."""
        import asyncio
        from src.core.models import ShiftQuery
        from src.tools.shift_tools import search_shifts
        query = ShiftQuery(agency_id=agency_id, date_from=date_from, date_to=date_to)
        result = asyncio.run(search_shifts(query))
        return json.dumps(result[:10], indent=2)

    @crewai_tool("GetSchedule")
    def get_schedule_tool(agency_id: int, date: str) -> str:
        """Get the full daily schedule for an agency."""
        import asyncio
        from src.tools.shift_tools import get_schedule
        result = asyncio.run(get_schedule(agency_id, date))
        return json.dumps(result, indent=2)

    return [analyze_coverage_tool, search_shifts_tool, get_schedule_tool]


# ── Crew builder ──────────────────────────────────────────────────────────────

def build_scheduling_crew(
    agency_id: int,
    date_from: str,
    date_to: str,
    verbose: bool = False,
) -> Any:
    """Build a 3-agent sequential crew for scheduling analysis."""
    from crewai import Agent, Crew, LLM, Process, Task

    settings = get_settings()
    tools = _build_crewai_tools()

    llm = LLM(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
        temperature=0.0,
        max_tokens=1024,
    )

    # ── Agents ────────────────────────────────────────────────────────────────

    analyst = Agent(
        role="Coverage Analyst",
        goal=(
            "Collect and analyze shift coverage data for emergency services agencies. "
            "Identify staffing gaps and compute coverage percentages."
        ),
        backstory=(
            "You are a data analyst specializing in emergency services scheduling. "
            "You use quantitative methods to assess staffing levels."
        ),
        tools=tools,
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=3,
    )

    strategist = Agent(
        role="Risk Strategist",
        goal=(
            "Evaluate staffing risks and produce a prioritized action plan "
            "based on coverage analysis data."
        ),
        backstory=(
            "You are a risk management expert for public safety operations. "
            "You translate data into actionable mitigation strategies."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=2,
    )

    reporter = Agent(
        role="Executive Report Writer",
        goal=(
            "Synthesize analysis and strategy into a clear, concise executive briefing "
            "suitable for fire chiefs and EMS directors."
        ),
        backstory=(
            "You are a senior communications specialist for public safety agencies. "
            "You write crisp, decision-ready reports for leadership."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=2,
    )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    analysis_task = Task(
        description=(
            f"Analyze shift coverage for agency {agency_id} from {date_from} to {date_to}. "
            "Use the AnalyzeCoverage and SearchShifts tools to collect data. "
            "Report: total shifts, filled shifts, coverage %, risk level, and top 5 gaps."
        ),
        expected_output=(
            "A structured JSON object with keys: agency_id, coverage_pct, risk_level, "
            "total_shifts, filled_shifts, critical_gaps (list), open_shifts_count."
        ),
        agent=analyst,
        tools=tools,
    )

    strategy_task = Task(
        description=(
            "Based on the coverage analysis, develop a risk mitigation action plan. "
            "Prioritize by severity. Include immediate actions (0-48h), "
            "short-term (1 week), and long-term (1 month)."
        ),
        expected_output=(
            "A structured plan with three sections: immediate_actions, "
            "short_term_actions, long_term_actions. Each action should have "
            "priority (high/medium/low), description, and estimated_impact."
        ),
        agent=strategist,
        context=[analysis_task],
    )

    report_task = Task(
        description=(
            "Write a concise executive briefing (max 250 words) for fire/EMS leadership. "
            "Include: situation summary, key risks, recommended actions, and next review date. "
            "Tone: professional, direct, decision-focused."
        ),
        expected_output=(
            "A polished executive briefing in plain text. "
            "Sections: SITUATION, RISKS, RECOMMENDED ACTIONS, NEXT STEPS."
        ),
        agent=reporter,
        context=[analysis_task, strategy_task],
    )

    # ── Crew ──────────────────────────────────────────────────────────────────

    return Crew(
        agents=[analyst, strategist, reporter],
        tasks=[analysis_task, strategy_task, report_task],
        process=Process.sequential,
        verbose=verbose,
    )


# ── Mock execution (demo mode) ────────────────────────────────────────────────

async def _mock_crew_run(agency_id: int, date_from: str, date_to: str) -> dict[str, Any]:
    """Simulate a crew run without LLM calls — used in demo mode."""
    import asyncio
    from src.tools.shift_tools import analyze_coverage

    coverage = await analyze_coverage(agency_id, date_from, date_to)
    pct = coverage["coverage_pct"]
    risk = coverage["risk_level"]

    return {
        "agency_id": agency_id,
        "coverage_pct": pct,
        "risk_level": risk,
        "action_plan": {
            "immediate_actions": [
                {"priority": "high", "description": "Fill night-shift vacancies via on-call list", "estimated_impact": "Coverage +5%"},
            ],
            "short_term_actions": [
                {"priority": "medium", "description": "Post overtime opportunities for next week", "estimated_impact": "Coverage +8%"},
            ],
            "long_term_actions": [
                {"priority": "low", "description": "Initiate recruitment campaign for 3 Paramedic positions", "estimated_impact": "Coverage +15%"},
            ],
        },
        "executive_briefing": (
            f"SITUATION\n"
            f"Agency {agency_id} coverage is {pct}% ({risk} risk) for {date_from}–{date_to}.\n\n"
            f"RISKS\n"
            f"• {len(coverage['critical_gaps'])} critical gaps identified\n"
            f"• Night-shift positions most affected\n\n"
            f"RECOMMENDED ACTIONS\n"
            f"• Immediate: activate on-call list for open shifts\n"
            f"• This week: post overtime opportunities\n"
            f"• This month: recruit 3 Paramedic positions\n\n"
            f"NEXT STEPS\n"
            f"Review coverage metrics daily. Escalate if coverage drops below 70%."
        ),
        "agents_used": ["Coverage Analyst", "Risk Strategist", "Executive Report Writer"],
        "tasks_completed": 3,
        "mode": "demo (mock — set ANTHROPIC_API_KEY for real LLM run)",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_crew_demo(
    agency_id: int = 42,
    date_from: str = "2025-01-01",
    date_to: str = "2025-01-31",
) -> dict[str, Any]:
    """Run the scheduling crew. Falls back to mock in demo mode."""
    settings = get_settings()

    if settings.demo_mode:
        log.warning("crewai_demo_mock_mode", reason="no ANTHROPIC_API_KEY")
        return await _mock_crew_run(agency_id, date_from, date_to)

    import asyncio

    crew = build_scheduling_crew(agency_id, date_from, date_to, verbose=False)

    # CrewAI kickoff is synchronous — run in thread to avoid blocking event loop
    from src.core.concurrency import run_in_thread

    def _kickoff() -> Any:
        return crew.kickoff()

    crew_output = await run_in_thread(_kickoff)

    return {
        "agency_id": agency_id,
        "executive_briefing": str(crew_output),
        "tasks_completed": len(crew.tasks),
        "agents_used": [a.role for a in crew.agents],
        "mode": "live (LLM)",
    }
