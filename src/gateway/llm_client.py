"""Unified LLM client with multi-provider routing and observability.

Demonstrates:
- LiteLLM for provider-agnostic calls + automatic fallback
- Helicone header injection (zero-code observability)
- Cost tracking per call
- Semaphore-limited concurrency
- tenacity retry with LLM-specific exceptions
"""

from __future__ import annotations

import time
from typing import Any

import anthropic
import litellm
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import get_settings
from src.core.concurrency import LLMSemaphore
from src.core.models import AgentResult

log = structlog.get_logger(__name__)

# ── Global semaphore ──────────────────────────────────────────────────────────

_sem = LLMSemaphore(max_concurrent=get_settings().max_concurrent_tools)


# ── LLM Gateway client ────────────────────────────────────────────────────────


class LLMClient:
    """Thin wrapper around LiteLLM adding retry, cost tracking and Helicone."""

    PRIMARY_MODEL = "anthropic/claude-haiku-4-5"
    FALLBACK_MODEL = "openai/gpt-4o-mini"

    def __init__(self) -> None:
        self._settings = get_settings()
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        litellm.set_verbose = self._settings.app_env == "development"
        # Suppress noisy httpx logs in tests
        litellm.suppress_debug_info = True

    def _helicone_headers(self, **tags: str) -> dict[str, str]:
        """Inject Helicone observability headers if key is configured."""
        if not self._settings.helicone_api_key:
            return {}
        return {
            "Helicone-Auth": f"Bearer {self._settings.helicone_api_key}",
            **{f"Helicone-Property-{k.title()}": v for k, v in tags.items()},
        }

    @_sem.guard
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(
            (anthropic.RateLimitError, anthropic.APIConnectionError)
        ),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        agency_id: int = 0,
        feature: str = "default",
    ) -> AgentResult[str]:
        """Call the LLM with fallback, retry and cost tracking.

        Returns an AgentResult[str] containing the text response plus
        token counts and USD cost.
        """
        chosen_model = model or self.PRIMARY_MODEL
        extra_headers = self._helicone_headers(agency_id=str(agency_id), feature=feature)

        start = time.perf_counter()
        response = await litellm.acompletion(
            model=chosen_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            fallbacks=[self.FALLBACK_MODEL],
            extra_headers=extra_headers or None,
            api_key=self._settings.anthropic_api_key or None,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content or ""
        usage = response.usage or {}
        cost = litellm.completion_cost(completion_response=response)

        log.info(
            "llm_call",
            model=response.model,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            cost_usd=round(cost, 6),
            duration_ms=round(duration_ms, 1),
        )

        return AgentResult(
            data=text,
            model_used=response.model,
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            cost_usd=cost,
            duration_ms=duration_ms,
        )


# ── Demo mode fallback (no API key needed) ────────────────────────────────────


class MockLLMClient:
    """In-memory mock — used in demo_mode and tests."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AgentResult[str]:
        import json

        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        mock_data = {
            "agency_id": kwargs.get("agency_id", 1),
            "date_range": "2025-01-01/2025-01-31",
            "total_shifts": 120,
            "filled_shifts": 102,
            "coverage_pct": 85.0,
            "critical_gaps": ["Night shift Jan 15"],
            "recommendations": ["Recruit 2 night-shift staff"],
            "risk_level": "medium",
        }
        return AgentResult(
            data=json.dumps(mock_data),
            model_used="mock",
            input_tokens=len(last_user) // 4,
            output_tokens=50,
            cost_usd=0.0,
            duration_ms=10.0,
        )


def get_llm_client() -> LLMClient | MockLLMClient:
    settings = get_settings()
    if settings.demo_mode:
        log.warning("demo_mode_active", reason="no ANTHROPIC_API_KEY configured")
        return MockLLMClient()
    return LLMClient()
