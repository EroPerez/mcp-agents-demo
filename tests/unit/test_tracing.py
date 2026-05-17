"""Unit tests for OpenTelemetry tracing utilities."""

from __future__ import annotations

import pytest
from opentelemetry import trace


@pytest.fixture(autouse=True)
def reset_tracer():
    """Reset OTel provider between tests."""
    from src.core import tracing
    tracing._tracer_provider = None
    trace.set_tracer_provider(trace.NoOpTracerProvider())
    yield
    tracing._tracer_provider = None


def test_configure_tracing_dev_mode(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    from src.core.config import get_settings
    get_settings.cache_clear()
    from src.core.tracing import configure_tracing
    configure_tracing()  # should not raise
    get_settings.cache_clear()


def test_get_tracer_returns_tracer():
    from src.core.tracing import configure_tracing, get_tracer
    configure_tracing()
    tracer = get_tracer("test.module")
    assert tracer is not None


@pytest.mark.asyncio
async def test_traced_decorator_success():
    from src.core.tracing import configure_tracing, traced
    configure_tracing()

    @traced("test.span")
    async def my_fn(x: int) -> int:
        return x * 2

    result = await my_fn(21)
    assert result == 42


@pytest.mark.asyncio
async def test_traced_decorator_records_exception():
    from src.core.tracing import configure_tracing, traced
    configure_tracing()

    @traced("test.error_span")
    async def failing_fn() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await failing_fn()


def test_add_span_attributes_no_crash():
    from src.core.tracing import add_span_attributes, configure_tracing
    configure_tracing()
    # Outside a span — should not raise
    add_span_attributes(agency_id=42, feature="test")
