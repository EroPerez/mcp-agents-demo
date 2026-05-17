# mcp-agents-demo

> **Production scaffold** for MCP servers, AI agents and orchestration in Python.  
> Every advanced feature from the guide — decorators, Pydantic v2, asyncio, ThreadPoolExecutor, ProcessPoolExecutor, FastMCP, Pydantic AI, LiteLLM, structlog — working together in one repo.

[![CI](https://github.com/EroPerez/mcp-agents-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/EroPerez/mcp-agents-demo/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Features

| Layer | Tool | Demo file |
|---|---|---|
| MCP Server | FastMCP 2.x | `src/server/main.py` |
| Validation | Pydantic v2 (discriminated unions, generics) | `src/core/models.py` |
| AI Agent | Pydantic AI (structured output) | `src/agents/coverage_agent.py` |
| Async concurrency | asyncio · Semaphore · Queue · TaskGroup | `src/core/concurrency.py` |
| Blocking I/O | ThreadPoolExecutor + contextvars | `src/core/concurrency.py` |
| CPU-bound | ProcessPoolExecutor (no GIL) | `src/core/concurrency.py` |
| Decorators | factory · class-based · registry · stacking | `src/core/decorators.py` |
| LLM Gateway | LiteLLM (fallback + cost) + Helicone | `src/gateway/llm_client.py` |
| Observability | structlog JSON + tenacity retry | `src/core/logging.py` |
| Settings | pydantic-settings (.env + env vars) | `src/core/config.py` |
| Testing | pytest-asyncio + AsyncMock | `tests/` |
| CI/CD | GitHub Actions | `.github/workflows/ci.yml` |

---

## Project Structure

```
mcp-agents-demo/
├── src/
│   ├── core/
│   │   ├── config.py          # BaseSettings — single source of truth
│   │   ├── concurrency.py     # asyncio · ThreadPool · ProcessPool · Queue
│   │   ├── decorators.py      # @tool · @timed · RateLimit · llm_retry
│   │   ├── logging.py         # structlog JSON structured logging
│   │   └── models.py          # Pydantic v2 models + discriminated unions
│   ├── gateway/
│   │   └── llm_client.py      # LiteLLM + Helicone unified client
│   ├── tools/
│   │   └── shift_tools.py     # Domain tools with @tool decorator
│   ├── agents/
│   │   ├── coverage_agent.py  # Pydantic AI structured-output agent
│   │   └── runner.py          # Multi-demo orchestration entry point
│   └── server/
│       └── main.py            # FastMCP server (tools + resources + prompts)
├── tests/
│   ├── unit/                  # Fast, no-network unit tests
│   └── integration/           # Agent tests in demo mode (no API key needed)
├── .github/workflows/ci.yml
├── pyproject.toml             # uv / hatchling / ruff / mypy / pytest config
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 1. Clone and install

```bash
git clone https://github.com/EroPerez/mcp-agents-demo
cd mcp-agents-demo
uv sync
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY
# Leave it empty to run in demo mode (no API calls)
```

### 3. Run the demos

```bash
# Demo mode (no API key needed) — shows all concurrency patterns
uv run demo-agents

# With a real API key — uses Claude + Pydantic AI agent
ANTHROPIC_API_KEY=sk-ant-... uv run demo-agents
```

### 4. Run the MCP server

```bash
# stdio transport (Claude Desktop)
uv run mcp-server

# SSE transport (HTTP, for production / Claude.ai)
MCP_TRANSPORT=sse uv run mcp-server
```

### 5. Run tests

```bash
# All tests (demo mode — no API key required)
uv run pytest

# Unit tests only (fastest)
uv run pytest tests/unit/ -v

# With coverage report
uv run pytest --cov=src --cov-report=html
```

---

## Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "scheduling-demo": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-agents-demo", "mcp-server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

Available tools in Claude Desktop:

- `tool_search_shifts` — find open shifts
- `tool_get_schedule` — daily schedule view
- `tool_update_shift_status` — mutate shift status
- `tool_analyze_coverage` — direct coverage stats
- `tool_ai_analyze_coverage` — LLM-powered deep analysis

---

## Key Patterns Explained

### Decorator stacking (order matters)

```python
@timed                          # 3rd — measures total time incl. retry
@llm_retry(max_attempts=4)      # 2nd — retries on RateLimitError
@RateLimit(calls_per_second=5)  # 1st — closest to function, limits call rate
async def call_llm(prompt: str) -> str: ...
```

### Discriminated Union (O(1) dispatch)

```python
Content = Annotated[
    Union[TextContent, ToolUseContent, ToolResultContent],
    Field(discriminator="type"),  # Pydantic picks the right model by "type" field
]
```

### Semaphore-limited gather

```python
results = await gather_with_limit(
    [agent.run(aid, ...) for aid in agency_ids],
    max_concurrent=3,  # at most 3 concurrent LLM calls
)
```

### Blocking I/O in async (context propagation)

```python
# contextvars are copied to the thread — request_id is preserved
result = await run_in_thread(legacy_sync_db_call, params)
```

---

## LLM Gateway

The `LLMClient` in `src/gateway/llm_client.py` provides:

- **LiteLLM** — single API for Anthropic, OpenAI, Bedrock, etc.
- **Automatic fallback** — if Claude fails, falls back to GPT-4o-mini
- **Cost tracking** — `litellm.completion_cost()` per call
- **Helicone** — add `HELICONE_API_KEY` to get dashboard observability with zero code changes
- **Semaphore** — caps concurrent calls to `MAX_CONCURRENT_TOOLS`
- **tenacity retry** — exponential backoff on `RateLimitError` / `APIConnectionError`

---

## Docker

```bash
# Build and run MCP server via HTTP/SSE
docker compose up --build

# Server available at http://localhost:8000
```

---

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Pre-commit hooks (lint + format on every commit)
uv run pre-commit install
```

---

## Roadmap

- [ ] Redis-backed semantic cache layer
- [ ] Portkey guardrails integration
- [ ] OpenTelemetry tracing (Jaeger export)
- [ ] Multi-agent handoff (router → specialist pattern)
- [ ] LiteLLM Proxy server `docker-compose` service

---

## License

MIT © Erodis Pérez Michel
