"""Unit tests for the multi-agent router."""

from __future__ import annotations

import pytest

from src.agents.router_agent import RouterAgent, _classify_local


@pytest.mark.parametrize("query,expected_domain", [
    ("What open shifts are available next week?",    "scheduling"),
    ("Show me the schedule for agency 42",           "scheduling"),
    ("What is the coverage risk this month?",        "coverage"),
    ("How many gaps do we have in January?",         "coverage"),
    ("Generate an executive summary report",         "reporting"),
    ("Give me an overview of all agencies",          "reporting"),
    ("What is the weather like?",                    "unknown"),
])
def test_local_classifier(query: str, expected_domain: str):
    decision = _classify_local(query)
    assert decision.domain == expected_domain


@pytest.mark.asyncio
async def test_router_scheduling(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.core.config import get_settings
    get_settings.cache_clear()

    router = RouterAgent()
    result = await router.run("What shifts are available for agency 1?", agency_id=1)
    assert result.domain == "scheduling"
    assert len(result.answer) > 0
    assert len(result.tools_called) > 0


@pytest.mark.asyncio
async def test_router_coverage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.core.config import get_settings
    get_settings.cache_clear()

    router = RouterAgent()
    result = await router.run("Analyze coverage risk for agency 42", agency_id=42)
    assert result.domain == "coverage"
    assert "coverage" in result.answer.lower() or "%" in result.answer


@pytest.mark.asyncio
async def test_router_unknown(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.core.config import get_settings
    get_settings.cache_clear()

    router = RouterAgent()
    result = await router.run("What is the capital of France?", agency_id=1)
    assert result.domain == "unknown"


@pytest.mark.asyncio
async def test_router_batch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.core.config import get_settings
    get_settings.cache_clear()

    router = RouterAgent()
    queries = [
        ("search for shifts", 1),
        ("coverage analysis",  2),
        ("executive report",   3),
    ]
    results = await router.run_batch(queries, max_concurrent=2)
    assert len(results) == 3
    domains = [r.domain for r in results]
    assert "scheduling" in domains
    assert "coverage"   in domains
    assert "reporting"  in domains
