"""Domain models using Pydantic v2.

Demonstrates:
- BaseModel with field_validator and model_validator
- Discriminated Unions for agent message content
- Generics + TypeVar for type-safe Agent wrapper
- BaseModel JSON Schema export (for LLM tool definitions)
- Dataclass with slots=True for hot-path objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Generic, Literal, TypeVar, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Generic type params ───────────────────────────────────────────────────────

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


# ── Tool call models ──────────────────────────────────────────────────────────


class ToolCallRequest(BaseModel):
    """Validated incoming tool call from the LLM."""

    model_config = {"str_strip_whitespace": True}

    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, object] = Field(default_factory=dict)
    agency_id: Annotated[int, Field(gt=0)]
    request_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        allowed = {"search_shifts", "get_schedule", "update_shift_status", "analyze_coverage"}
        if v not in allowed:
            raise ValueError(f"Unknown tool: '{v}'. Allowed: {allowed}")
        return v

    @model_validator(mode="after")
    def require_args_for_mutations(self) -> "ToolCallRequest":
        mutation_tools = {"update_shift_status"}
        if self.tool_name in mutation_tools and not self.arguments:
            raise ValueError(f"'{self.tool_name}' requires non-empty arguments")
        return self


class ToolCallResult(BaseModel):
    tool_name: str
    success: bool
    data: object = None
    error: str | None = None
    duration_ms: float = 0.0


# ── Agent message content (discriminated union) ───────────────────────────────


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ToolUseContent(BaseModel):
    type: Literal["tool_use"]
    id: str
    tool_name: str
    input: dict[str, object]


class ToolResultContent(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str
    is_error: bool = False


# Pydantic selects the correct model in O(1) via the 'type' discriminator field
Content = Annotated[
    Union[TextContent, ToolUseContent, ToolResultContent],
    Field(discriminator="type"),
]


class AgentMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: list[Content]
    metadata: dict[str, object] = Field(default_factory=dict)


class ConversationHistory(BaseModel):
    messages: list[AgentMessage] = Field(default_factory=list)
    max_turns: int = Field(default=20, gt=0)

    def add(self, message: AgentMessage) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_turns * 2:
            # Keep system message + last N turns
            self.messages = self.messages[:1] + self.messages[-(self.max_turns * 2 - 1):]

    def to_api_format(self) -> list[dict]:
        return [m.model_dump(exclude_none=True) for m in self.messages]


# ── Scheduling domain models ──────────────────────────────────────────────────


class ShiftQuery(BaseModel):
    """Input schema for the search_shifts MCP tool.
    The JSON Schema is exported automatically for LLM tool definitions.
    """

    agency_id: int = Field(gt=0, description="Unique agency identifier")
    date_from: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Start date YYYY-MM-DD")
    date_to: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="End date YYYY-MM-DD")
    positions: list[str] = Field(default_factory=list, description="Filter by position codes")

    @model_validator(mode="after")
    def date_order(self) -> "ShiftQuery":
        if self.date_from > self.date_to:
            raise ValueError("date_from must be <= date_to")
        return self


class Shift(BaseModel):
    shift_id: int
    agency_id: int
    position: str
    date: str
    start_time: str
    end_time: str
    status: Literal["open", "filled", "cancelled"]
    assigned_to: str | None = None


class CoverageAnalysis(BaseModel):
    """Structured output produced by the coverage analysis agent."""

    agency_id: int
    date_range: str
    total_shifts: int
    filled_shifts: int
    coverage_pct: float = Field(ge=0.0, le=100.0)
    critical_gaps: list[str]
    recommendations: list[str]
    risk_level: Literal["low", "medium", "high", "critical"]


# ── Hot-path dataclass (slots + frozen for performance) ───────────────────────


@dataclass(slots=True, frozen=True)
class CacheKey:
    """Immutable, hashable cache key — safe as dict/set key."""

    agency_id: int
    date: str
    tool_name: str

    def __str__(self) -> str:
        return f"{self.tool_name}:{self.agency_id}:{self.date}"


# ── Generic AgentResult wrapper ───────────────────────────────────────────────


class AgentResult(BaseModel, Generic[OutputT]):
    """Type-safe wrapper around any structured agent output."""

    data: OutputT
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
