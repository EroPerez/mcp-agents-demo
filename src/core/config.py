"""Application configuration loaded from environment / .env file.

Usage:
    from src.core.config import get_settings

    settings = get_settings()
    print(settings.anthropic_api_key)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for all configuration.

    Values are loaded in this order (highest priority first):
    1. Environment variables
    2. .env file
    3. Field defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Providers ────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key (fallback)")

    # ── Gateway ──────────────────────────────────────────────────────────────
    litellm_master_key: str = Field(default="sk-demo")
    helicone_api_key: str = Field(default="")
    portkey_api_key: str = Field(default="")

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_concurrent_tools: int = Field(default=5, gt=0, le=50)
    agent_timeout_seconds: float = Field(default=30.0, gt=0)

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite+aiosqlite:///./demo.db")
    redis_url: str = Field(default="redis://localhost:6379")

    # ── MCP Server ────────────────────────────────────────────────────────────
    mcp_transport: Literal["stdio", "sse"] = "stdio"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = Field(default=8000, gt=0, lt=65536)

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_env(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def demo_mode(self) -> bool:
        """True when no real API keys are configured."""
        return not self.anthropic_api_key.startswith("sk-ant-")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached)."""
    return Settings()
