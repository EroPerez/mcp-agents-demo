"""LangChain Agent Demo.

Demonstrates:
- LCEL (LangChain Expression Language) chain composition with | operator
- Tool calling agent (create_tool_calling_agent + AgentExecutor)
- @tool decorator with Pydantic v2 args schema
- Streaming response with astream_events
- Runnable with retry + timeout
- MockLLM fallback for demo mode (no API key needed)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from src.core.config import get_settings
from src.core.models import ShiftQuery
from src.tools.shift_tools import analyze_coverage, get_schedule, search_shifts

log = structlog.get_logger(__name__)


# ── LangChain tools (thin wrappers over our domain tools) ─────────────────────


def _build_langchain_tools() -> list:
    """Build LangChain @tool wrappers lazily (avoids import cost at module level)."""
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    async def lc_search_shifts(agency_id: int, date_from: str, date_to: str) -> str:
        """Search open shifts for an agency within a date range.

        Args:
            agency_id: Agency identifier (positive integer).
            date_from: Start date in YYYY-MM-DD format.
            date_to: End date in YYYY-MM-DD format.
        """
        query = ShiftQuery(agency_id=agency_id, date_from=date_from, date_to=date_to)
        result = await search_shifts(query)
        return json.dumps(result[:10])  # cap for context window

    @lc_tool
    async def lc_get_schedule(agency_id: int, date: str) -> str:
        """Get the complete daily schedule for an agency.

        Args:
            agency_id: Agency identifier.
            date: Target date in YYYY-MM-DD format.
        """
        result = await get_schedule(agency_id, date)
        return json.dumps(result)

    @lc_tool
    async def lc_analyze_coverage(agency_id: int, date_from: str, date_to: str) -> str:
        """Analyze staffing coverage and return risk level + recommendations.

        Args:
            agency_id: Agency identifier.
            date_from: Start date YYYY-MM-DD.
            date_to: End date YYYY-MM-DD.
        """
        result = await analyze_coverage(agency_id, date_from, date_to)
        return json.dumps(result)

    return [lc_search_shifts, lc_get_schedule, lc_analyze_coverage]


# ── Demo 1: Simple LCEL chain (no tools) ─────────────────────────────────────


async def demo_lcel_chain(agency_id: int, date: str) -> str:
    """LCEL chain: prompt | llm | output_parser.

    The | operator composes Runnables — each step's output
    is the next step's input.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    settings = get_settings()

    schedule_data = await get_schedule(agency_id, date)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert in emergency services scheduling. "
            "Analyze the provided schedule data and give a concise briefing."
        )),
        ("human", (
            "Agency ID: {agency_id}\n"
            "Date: {date}\n"
            "Schedule data:\n{schedule}\n\n"
            "Provide a 3-bullet briefing for the shift supervisor."
        )),
    ])

    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
        max_tokens=512,
        temperature=0.0,
    ).with_retry(stop_after_attempt=3)

    # LCEL chain composition with | operator
    chain = (
        RunnablePassthrough()         # pass input dict through
        | prompt                      # render prompt template
        | llm                         # call LLM
        | StrOutputParser()           # extract text from AIMessage
    )

    result = await chain.ainvoke({
        "agency_id": agency_id,
        "date": date,
        "schedule": json.dumps(schedule_data, indent=2),
    })

    log.info("lcel_chain_done", agency_id=agency_id, date=date)
    return result


# ── Demo 2: Tool-calling agent ────────────────────────────────────────────────


async def demo_tool_calling_agent(query: str, agency_id: int = 42) -> dict[str, Any]:
    """Full tool-calling agent using create_tool_calling_agent + AgentExecutor.

    The agent decides which tools to call and in what order based on the query.
    """
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate

    settings = get_settings()
    tools = _build_langchain_tools()

    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
        max_tokens=1024,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert scheduling assistant for emergency services agencies. "
            "Use the provided tools to answer questions about shifts and coverage. "
            "Always use real data from the tools — never guess. "
            "Be concise and actionable."
        )),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
    )

    result = await executor.ainvoke({"input": query})

    log.info("tool_agent_done", query=query[:60])
    return {
        "query": query,
        "answer": result.get("output", ""),
        "intermediate_steps": len(result.get("intermediate_steps", [])),
    }


# ── Demo 3: Streaming with astream_events ────────────────────────────────────


async def demo_streaming(agency_id: int, date_from: str, date_to: str) -> str:
    """Stream LLM tokens using astream_events (v2 API).

    Shows how to handle streaming in async context —
    essential for real-time UIs and long-running analysis.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    settings = get_settings()

    coverage_data = await analyze_coverage(agency_id, date_from, date_to)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a staffing analyst. Be concise."),
        ("human", (
            "Coverage data for agency {agency_id} ({date_from} to {date_to}):\n"
            "{coverage}\n\n"
            "Write a 2-paragraph executive summary with key risks and actions."
        )),
    ])

    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
        max_tokens=512,
        temperature=0.0,
    )

    chain = prompt | llm | StrOutputParser()

    collected = []
    async for event in chain.astream_events(
        {
            "agency_id": agency_id,
            "date_from": date_from,
            "date_to": date_to,
            "coverage": json.dumps(coverage_data, indent=2),
        },
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if isinstance(chunk, str):
                collected.append(chunk)

    return "".join(collected)


# ── Demo 4: Parallel chains with gather ──────────────────────────────────────


async def demo_parallel_chains(agency_ids: list[int], date: str) -> list[dict]:
    """Run one LCEL chain per agency concurrently using asyncio.gather.

    Combines LangChain's async interface with asyncio concurrency patterns.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    settings = get_settings()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a scheduling assistant. Be very brief — one sentence only."),
        ("human", "Agency {agency_id} schedule summary for {date}: {schedule}"),
    ])

    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
        max_tokens=100,
        temperature=0.0,
    )
    chain = prompt | llm | StrOutputParser()

    async def run_one(agency_id: int) -> dict:
        schedule = await get_schedule(agency_id, date)
        summary = await chain.ainvoke({
            "agency_id": agency_id,
            "date": date,
            "schedule": json.dumps(schedule),
        })
        return {"agency_id": agency_id, "summary": summary}

    # All agencies run concurrently
    return list(await asyncio.gather(*[run_one(aid) for aid in agency_ids]))


# ── Mock fallback (demo mode) ─────────────────────────────────────────────────


async def demo_lcel_chain_mock(agency_id: int, date: str) -> str:
    coverage = await analyze_coverage(agency_id, date, date)
    return (
        f"• Coverage: {coverage['coverage_pct']}% ({coverage['risk_level']} risk)\n"
        f"• Filled: {coverage['filled_shifts']}/{coverage['total_shifts']} shifts\n"
        f"• Action: {coverage['recommendations'][0]}"
    )


async def demo_tool_calling_agent_mock(query: str, agency_id: int = 42) -> dict[str, Any]:
    coverage = await analyze_coverage(agency_id, "2025-01-01", "2025-01-07")
    return {
        "query": query,
        "answer": (
            f"Agency {agency_id} has {coverage['coverage_pct']}% coverage "
            f"({coverage['risk_level']} risk). "
            f"Recommendation: {coverage['recommendations'][0]}"
        ),
        "intermediate_steps": 2,
    }


# ── Unified entrypoint ────────────────────────────────────────────────────────


class LangChainDemos:
    """Runs all four LangChain demos, falling back to mock in demo mode."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def run_all(self) -> dict[str, Any]:
        if self._settings.demo_mode:
            log.warning("langchain_demo_mock_mode", reason="no ANTHROPIC_API_KEY")
            return await self._run_mock()
        return await self._run_real()

    async def _run_real(self) -> dict[str, Any]:
        lcel = await demo_lcel_chain(agency_id=42, date="2025-01-15")
        agent = await demo_tool_calling_agent(
            "What is the coverage risk for agency 42 during January 2025? "
            "Identify the top 3 staffing gaps.",
            agency_id=42,
        )
        streaming = await demo_streaming(
            agency_id=42, date_from="2025-01-01", date_to="2025-01-31"
        )
        parallel = await demo_parallel_chains(
            agency_ids=[10, 20, 30], date="2025-01-15"
        )
        return {
            "lcel_chain": lcel,
            "tool_agent": agent,
            "streaming": streaming,
            "parallel_chains": parallel,
        }

    async def _run_mock(self) -> dict[str, Any]:
        lcel = await demo_lcel_chain_mock(agency_id=42, date="2025-01-15")
        agent = await demo_tool_calling_agent_mock(
            "What is the coverage risk for agency 42?", agency_id=42
        )
        parallel = await asyncio.gather(*[
            demo_tool_calling_agent_mock("summary", aid) for aid in [10, 20, 30]
        ])
        return {
            "lcel_chain": lcel,
            "tool_agent": agent,
            "streaming": "(streaming skipped in demo mode)",
            "parallel_chains": [{"agency_id": r["query"], "summary": r["answer"]} for r in parallel],
        }
