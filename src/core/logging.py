"""Structured logging via structlog.

Call configure_logging() once at startup. After that, use:

    import structlog
    log = structlog.get_logger(__name__)
    log.info("event", key="value")
"""

from __future__ import annotations

import logging
import sys

import structlog

from src.core.config import get_settings


def configure_logging() -> None:
    """Configure structlog with JSON renderer for production, pretty for dev."""
    settings = get_settings()
    level = getattr(logging, settings.log_level)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)  # type: ignore[assignment]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def bind_request_context(request_id: str, **extra: object) -> None:
    """Bind values to the current async context (visible in all log calls)."""
    structlog.contextvars.bind_contextvars(request_id=request_id, **extra)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
