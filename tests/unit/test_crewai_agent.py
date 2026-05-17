"""Unit tests for the CrewAI crew demo (mock mode)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def force_demo_mode(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_crew_demo_mock_returns_expected_keys():
    from src.agents.crewai_agent import run_crew_demo

    result = await run_crew_demo(agency_id=1, date_from="2025-01-01", date_to="2025-01-07")

    assert "agency_id" in result
    assert "executive_briefing" in result
    assert "agents_used" in result
    assert "tasks_completed" in result
    assert result["agency_id"] == 1


@pytest.mark.asyncio
async def test_crew_demo_mock_agents_present():
    from src.agents.crewai_agent import run_crew_demo

    result = await run_crew_demo(agency_id=42, date_from="2025-01-01", date_to="2025-01-31")
    agents = result["agents_used"]

    assert "Coverage Analyst" in agents
    assert "Risk Strategist" in agents
    assert "Executive Report Writer" in agents
    assert result["tasks_completed"] == 3


@pytest.mark.asyncio
async def test_crew_demo_mock_briefing_content():
    from src.agents.crewai_agent import run_crew_demo

    result = await run_crew_demo(agency_id=10, date_from="2025-02-01", date_to="2025-02-28")
    briefing = result["executive_briefing"]

    assert "SITUATION" in briefing
    assert "RISKS" in briefing
    assert "RECOMMENDED ACTIONS" in briefing
    assert "NEXT STEPS" in briefing


@pytest.mark.asyncio
async def test_crew_mock_run_directly():
    from src.agents.crewai_agent import _mock_crew_run

    result = await _mock_crew_run(5, "2025-01-01", "2025-01-07")
    assert 0 <= result["coverage_pct"] <= 100
    assert result["risk_level"] in ("low", "medium", "high", "critical")
    assert "immediate_actions" in result["action_plan"]
