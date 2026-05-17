"""Integration tests for the Coverage Agent (demo mode — no real API key)."""

from __future__ import annotations

import pytest

from src.agents.coverage_agent import CoverageAgent, analyze_multiple_agencies
from src.core.models import AgentResult, CoverageAnalysis


@pytest.mark.asyncio
async def test_coverage_agent_demo_mode(monkeypatch):
    """Agent should run fully in demo mode without an API key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # force demo mode

    # Reload settings singleton
    from src.core import config
    config.get_settings.cache_clear()

    agent = CoverageAgent()
    result = await agent.run(agency_id=1, date_from="2025-01-01", date_to="2025-01-07")

    assert isinstance(result, AgentResult)
    assert isinstance(result.data, CoverageAnalysis)
    assert 0 <= result.data.coverage_pct <= 100
    assert result.data.risk_level in ("low", "medium", "high", "critical")
    assert result.data.total_shifts > 0


@pytest.mark.asyncio
async def test_coverage_analysis_values(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.core import config
    config.get_settings.cache_clear()

    agent = CoverageAgent()
    result = await agent.run(agency_id=42, date_from="2025-03-01", date_to="2025-03-31")
    analysis = result.data

    assert analysis.agency_id == 42
    assert analysis.filled_shifts <= analysis.total_shifts
    assert isinstance(analysis.critical_gaps, list)
    assert isinstance(analysis.recommendations, list)
    assert len(analysis.recommendations) >= 1


@pytest.mark.asyncio
async def test_multi_agency_parallel(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.core import config
    config.get_settings.cache_clear()

    results = await analyze_multiple_agencies(
        agency_ids=[1, 2, 3],
        date_from="2025-01-01",
        date_to="2025-01-03",
        max_concurrent=2,
    )
    assert len(results) == 3
    agency_ids = {r.data.agency_id for r in results}
    assert agency_ids == {1, 2, 3}
