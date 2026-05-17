"""OpenTelemetry distributed tracing.

Exports spans to:
- Jaeger  (OTLP gRPC, default: localhost:4317)
- stdout  (dev mode — no collector needed)

Usage:
    configure_tracing()          # call once at startup

    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("agency.id", 42)
        result = await do_work()

Or via decorator:
    @traced("search_shifts")
    async def search_shifts(query): ...
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

from src.core.config import get_settings

log = structlog.get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])

_tracer_provider: TracerProvider | None = None


def configure_tracing() -> None:
    """Initialize the global TracerProvider.

    - Production: exports to Jaeger via OTLP gRPC
    - Development: prints spans to stdout (no collector needed)
    """
    global _tracer_provider

    settings = get_settings()
    resource = Resource.create({
        "service.name":    "mcp-agents-demo",
        "service.version": "0.1.0",
        "deployment.environment": settings.app_env,
    })

    provider = TracerProvider(resource=resource)

    if settings.is_production:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_exporter = OTLPSpanExporter(
                endpoint="http://localhost:4317",  # override via OTEL_EXPORTER_OTLP_ENDPOINT
                insecure=True,
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            log.info("tracing_configured", exporter="otlp_grpc")
        except Exception as exc:
            log.warning("otlp_exporter_failed", error=str(exc), fallback="console")
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # Dev: print spans to stdout — no Jaeger needed
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        log.info("tracing_configured", exporter="console")

    trace.set_tracer_provider(provider)
    _tracer_provider = provider


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def traced(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator: wraps an async function in an OTel span.

    Usage:
        @traced("llm.call", attributes={"model": "claude-haiku-4-5"})
        async def call_llm(prompt: str) -> str: ...
    """
    def decorator(func: F) -> F:
        name = span_name or func.__name__
        tracer = get_tracer(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, str(v))
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def add_span_attributes(**kwargs: Any) -> None:
    """Add attributes to the current active span."""
    span = trace.get_current_span()
    if span.is_recording():
        for k, v in kwargs.items():
            span.set_attribute(k, str(v))


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider. Call on app exit."""
    if _tracer_provider:
        _tracer_provider.shutdown()
