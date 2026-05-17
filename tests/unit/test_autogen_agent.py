"""Unit tests for the AutoGen multi-agent demo (mock mode)."""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def force_demo_mode(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_autogen_demo_mock_structure():
    from src.agents.autogen_agent import run_autogen_demo

    result = await run_autogen_demo(agency_id=1, date_from="2025-01-01", date_to="2025-01-07")

    assert result["team"] == "RoundRobinGroupChat"
    assert result["agency_id"] == 1
    assert result["messages"] == 3
    assert len(result["conversation"]) == 3


@pytest.mark.asyncio
async def test_autogen_demo_agent_names():
    from src.agents.autogen_agent import run_autogen_demo

    result = await run_autogen_demo(agency_id=42, date_from="2025-01-01", date_to="2025-01-31")
    agents = [turn["agent"] for turn in result["conversation"]]

    assert agents[0] == "DataAgent"
    assert agents[1] == "AnalystAgent"
    assert agents[2] == "PlannerAgent"


@pytest.mark.asyncio
async def test_autogen_plan_complete_marker():
    from src.agents.autogen_agent import run_autogen_demo

    result = await run_autogen_demo(agency_id=5, date_from="2025-03-01", date_to="2025-03-31")
    last_message = result["conversation"][-1]["message"]
    assert "PLAN_COMPLETE" in last_message


@pytest.mark.asyncio
async def test_tool_fetch_coverage_data():
    from src.agents.autogen_agent import fetch_coverage_data

    raw = await fetch_coverage_data(1, "2025-01-01", "2025-01-07")
    data = json.loads(raw)
    assert "coverage_pct" in data
    assert "risk_level" in data


@pytest.mark.asyncio
async def test_tool_fetch_open_shifts():
    from src.agents.autogen_agent import fetch_open_shifts

    raw = await fetch_open_shifts(1, "2025-01-01", "2025-01-07")
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) <= 10


@pytest.mark.asyncio
async def test_tool_compute_risk_score():
    from src.agents.autogen_agent import compute_risk_score

    # High coverage → low risk
    raw = await compute_risk_score(95.0, 2)
    data = json.loads(raw)
    assert data["severity"] == "low"
    assert data["risk_score"] < 20

    # Low coverage → high/critical risk
    raw2 = await compute_risk_score(40.0, 20)
    data2 = json.loads(raw2)
    assert data2["severity"] in ("high", "critical")
    assert data2["risk_score"] > 40
