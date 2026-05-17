"""Unit tests for the LangChain agent demo (mock mode — no API key)."""

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
async def test_langchain_demos_mock_mode():
    from src.agents.langchain_agent import LangChainDemos

    results = await LangChainDemos().run_all()

    assert "lcel_chain" in results
    assert "tool_agent" in results
    assert "streaming" in results
    assert "parallel_chains" in results


@pytest.mark.asyncio
async def test_lcel_chain_mock_returns_string():
    from src.agents.langchain_agent import demo_lcel_chain_mock

    result = await demo_lcel_chain_mock(agency_id=42, date="2025-01-15")
    assert isinstance(result, str)
    assert "Coverage" in result


@pytest.mark.asyncio
async def test_tool_agent_mock_structure():
    from src.agents.langchain_agent import demo_tool_calling_agent_mock

    result = await demo_tool_calling_agent_mock("coverage risk agency 42", agency_id=42)
    assert "query" in result
    assert "answer" in result
    assert "intermediate_steps" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


@pytest.mark.asyncio
async def test_parallel_chains_mock_count():
    """Three agencies should return three results."""
    from src.agents.langchain_agent import LangChainDemos

    demos = LangChainDemos()
    results = await demos._run_mock()
    assert len(results["parallel_chains"]) == 3
