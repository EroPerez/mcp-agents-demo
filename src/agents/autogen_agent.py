"""AutoGen Multi-Agent Demo (autogen-agentchat 0.7+).

Demonstrates:
- AssistantAgent with registered function tools
- RoundRobinGroupChat — agents take turns responding
- SelectorGroupChat — LLM selects the next speaker
- MaxMessageTermination + TextMentionTermination conditions
- Console streaming for real-time output
- Mock execution in demo mode (no API key needed)

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  RoundRobinGroupChat                                 │
    │                                                      │
    │   DataAgent         AnalystAgent      PlannerAgent  │
    │   (fetches data)    (interprets)      (action plan) │
    │        │                 │                 │         │
    │        └────────── round-robin ────────────┘         │
    └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from src.core.config import get_settings

log = structlog.get_logger(__name__)


# ── Tool functions (must be top-level for autogen registration) ───────────────

async def fetch_coverage_data(agency_id: int, date_from: str, date_to: str) -> str:
    """Fetch coverage analysis data for an agency and date range.

    Args:
        agency_id: Agency identifier.
        date_from: Start date YYYY-MM-DD.
        date_to: End date YYYY-MM-DD.

    Returns:
        JSON string with coverage statistics.
    """
    from src.tools.shift_tools import analyze_coverage
    result = await analyze_coverage(agency_id, date_from, date_to)
    return json.dumps(result, indent=2)


async def fetch_open_shifts(agency_id: int, date_from: str, date_to: str) -> str:
    """Fetch list of open (unfilled) shifts for an agency.

    Args:
        agency_id: Agency identifier.
        date_from: Start date YYYY-MM-DD.
        date_to: End date YYYY-MM-DD.

    Returns:
        JSON string with list of open shifts (max 10).
    """
    from src.core.models import ShiftQuery
    from src.tools.shift_tools import search_shifts
    query = ShiftQuery(agency_id=agency_id, date_from=date_from, date_to=date_to)
    result = await search_shifts(query)
    return json.dumps(result[:10], indent=2)


async def compute_risk_score(coverage_pct: float, open_shifts: int) -> str:
    """Compute a composite risk score from coverage metrics.

    Args:
        coverage_pct: Coverage percentage (0-100).
        open_shifts: Number of unfilled shifts.

    Returns:
        JSON with risk_score (0-100), severity, and recommended_action.
    """
    base_score = 100 - coverage_pct
    penalty = min(open_shifts * 2, 20)
    score = min(base_score + penalty, 100)

    if score < 15:
        severity, action = "low", "Monitor weekly"
    elif score < 35:
        severity, action = "medium", "Review staffing plan this week"
    elif score < 60:
        severity, action = "high", "Activate on-call list immediately"
    else:
        severity, action = "critical", "Declare staffing emergency — notify leadership"

    return json.dumps({
        "risk_score": round(score, 1),
        "severity": severity,
        "recommended_action": action,
        "inputs": {"coverage_pct": coverage_pct, "open_shifts": open_shifts},
    }, indent=2)


# ── Agent builder ─────────────────────────────────────────────────────────────

def _make_model_client() -> Any:
    """Build an Anthropic model client for autogen-ext."""
    from autogen_ext.models.anthropic import AnthropicChatCompletionClient
    settings = get_settings()
    return AnthropicChatCompletionClient(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
        max_tokens=1024,
    )


def build_autogen_team() -> Any:
    """Build a RoundRobinGroupChat team with 3 specialized agents."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat

    model_client = _make_model_client()

    data_agent = AssistantAgent(
        name="DataAgent",
        model_client=model_client,
        tools=[fetch_coverage_data, fetch_open_shifts],
        system_message=(
            "You are a data collection agent. Your job is to use the provided tools "
            "to fetch coverage and shift data for the requested agency. "
            "Call the tools, report the raw numbers, and pass them to the next agent. "
            "Be concise — data only, no analysis."
        ),
        description="Fetches raw scheduling and coverage data using tools.",
    )

    analyst_agent = AssistantAgent(
        name="AnalystAgent",
        model_client=model_client,
        tools=[compute_risk_score],
        system_message=(
            "You are a data analyst. Interpret the coverage data provided by DataAgent. "
            "Use compute_risk_score to calculate the composite risk score. "
            "Identify the top 3 risk factors and their business impact."
        ),
        description="Interprets data and computes risk scores.",
    )

    planner_agent = AssistantAgent(
        name="PlannerAgent",
        model_client=model_client,
        system_message=(
            "You are an operations planner for emergency services. "
            "Based on the analysis from AnalystAgent, produce a concrete action plan "
            "with 3 prioritized recommendations. End your response with 'PLAN_COMPLETE'."
        ),
        description="Produces the final action plan.",
    )

    termination = (
        TextMentionTermination("PLAN_COMPLETE")
        | MaxMessageTermination(max_messages=9)
    )

    return RoundRobinGroupChat(
        participants=[data_agent, analyst_agent, planner_agent],
        termination_condition=termination,
    )


def build_selector_team() -> Any:
    """SelectorGroupChat — LLM chooses the next speaker based on context."""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import SelectorGroupChat

    model_client = _make_model_client()

    scheduler = AssistantAgent(
        name="Scheduler",
        model_client=model_client,
        tools=[fetch_open_shifts],
        system_message="You find and list open shifts for the requested agency and date range.",
        description="Specialist in shift availability queries.",
    )

    risk_assessor = AssistantAgent(
        name="RiskAssessor",
        model_client=model_client,
        tools=[fetch_coverage_data, compute_risk_score],
        system_message="You assess staffing risk using coverage data and risk scoring.",
        description="Specialist in risk assessment and mitigation.",
    )

    coordinator = AssistantAgent(
        name="Coordinator",
        model_client=model_client,
        system_message=(
            "You coordinate the final response. Summarize what the other agents found "
            "and provide a clear executive summary. Reply DONE when finished."
        ),
        description="Synthesizes outputs into a final executive summary.",
    )

    return SelectorGroupChat(
        participants=[scheduler, risk_assessor, coordinator],
        model_client=model_client,
        termination_condition=MaxMessageTermination(max_messages=6),
        selector_prompt=(
            "Select the most appropriate agent to respond next based on the conversation:\n"
            "{participants}\n\nConversation:\n{history}\n\nNext agent:"
        ),
    )


# ── Mock execution ────────────────────────────────────────────────────────────

async def _mock_team_run(agency_id: int, date_from: str, date_to: str) -> dict[str, Any]:
    """Simulate a full team conversation without LLM calls."""
    from src.tools.shift_tools import analyze_coverage
    from src.core.models import ShiftQuery
    from src.tools.shift_tools import search_shifts

    coverage = await analyze_coverage(agency_id, date_from, date_to)
    query = ShiftQuery(agency_id=agency_id, date_from=date_from, date_to=date_to)
    shifts = await search_shifts(query)

    risk_data = json.loads(await compute_risk_score(coverage["coverage_pct"], len(shifts)))

    conversation = [
        {
            "agent": "DataAgent",
            "message": (
                f"Fetched data for agency {agency_id}:\n"
                f"Coverage: {coverage['coverage_pct']}% | "
                f"Filled: {coverage['filled_shifts']}/{coverage['total_shifts']} | "
                f"Open shifts: {len(shifts)}"
            ),
        },
        {
            "agent": "AnalystAgent",
            "message": (
                f"Risk score: {risk_data['risk_score']}/100 ({risk_data['severity']} severity)\n"
                f"Top risks: understaffed nights, {len(coverage['critical_gaps'])} gaps identified\n"
                f"Recommended action: {risk_data['recommended_action']}"
            ),
        },
        {
            "agent": "PlannerAgent",
            "message": (
                f"Action Plan:\n"
                f"1. [HIGH] {risk_data['recommended_action']}\n"
                f"2. [MED] Post overtime for positions: {', '.join(p for p in ['Paramedic', 'EMT']) }\n"
                f"3. [LOW] Schedule weekly coverage review meeting\n\nPLAN_COMPLETE"
            ),
        },
    ]

    return {
        "team": "RoundRobinGroupChat",
        "agency_id": agency_id,
        "messages": len(conversation),
        "conversation": conversation,
        "final_risk": risk_data,
        "mode": "demo (mock — set ANTHROPIC_API_KEY for real LLM run)",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_autogen_demo(
    agency_id: int = 42,
    date_from: str = "2025-01-01",
    date_to: str = "2025-01-31",
) -> dict[str, Any]:
    """Run the AutoGen team. Falls back to mock in demo mode."""
    settings = get_settings()

    if settings.demo_mode:
        log.warning("autogen_demo_mock_mode", reason="no ANTHROPIC_API_KEY")
        return await _mock_team_run(agency_id, date_from, date_to)

    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken

    team = build_autogen_team()
    task = (
        f"Analyze staffing coverage for agency {agency_id} "
        f"from {date_from} to {date_to}. "
        "Collect data, assess risk, and produce an action plan."
    )

    messages = []
    async for msg in team.run_stream(
        task=task,
        cancellation_token=CancellationToken(),
    ):
        if hasattr(msg, "source") and hasattr(msg, "content"):
            messages.append({"agent": msg.source, "message": str(msg.content)})
            log.debug("autogen_message", agent=msg.source, chars=len(str(msg.content)))

    return {
        "team": "RoundRobinGroupChat",
        "agency_id": agency_id,
        "messages": len(messages),
        "conversation": messages,
        "mode": "live (LLM)",
    }
