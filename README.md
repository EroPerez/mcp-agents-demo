# mcp-agents-demo

> **Production scaffold** for MCP servers, AI agents and orchestration in Python.  
> Every advanced pattern — decorators, Pydantic v2, asyncio, ThreadPoolExecutor, ProcessPoolExecutor, FastMCP, Pydantic AI, LangChain, LiteLLM, Helicone, Portkey, Redis cache, OpenTelemetry — working together in one repo.

[![CI](https://github.com/EroPerez/mcp-agents-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/EroPerez/mcp-agents-demo/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Features

| Layer | Tool | File |
|---|---|---|
| MCP Server | FastMCP 2.x — tools, resources, prompts | `src/server/main.py` |
| Validation | Pydantic v2 — discriminated unions, generics | `src/core/models.py` |
| AI Agent | Pydantic AI — structured output | `src/agents/coverage_agent.py` |
| LangChain | LCEL · Tool Agent · Streaming · Parallel | `src/agents/langchain_agent.py` |
| Multi-Agent | Router → Specialist handoff pattern | `src/agents/router_agent.py` |
| Async concurrency | asyncio · Semaphore · Queue · TaskGroup | `src/core/concurrency.py` |
| Blocking I/O | ThreadPoolExecutor + contextvars | `src/core/concurrency.py` |
| CPU-bound | ProcessPoolExecutor — no GIL | `src/core/concurrency.py` |
| Decorators | factory · class-based · registry · stacking | `src/core/decorators.py` |
| LLM Gateway | LiteLLM — fallback routing + cost tracking | `src/gateway/llm_client.py` |
| Portkey | Guardrails · semantic cache · multi-provider | `src/gateway/portkey_client.py` |
| Cache | Redis + in-memory fallback, TTL per namespace | `src/core/cache.py` |
| Tracing | OpenTelemetry — `@traced` decorator, Jaeger export | `src/core/tracing.py` |
| Observability | structlog JSON + tenacity retry | `src/core/logging.py` |
| Settings | pydantic-settings — .env + env vars | `src/core/config.py` |
| Testing | pytest-asyncio — 50 unit + integration tests | `tests/` |
| CI/CD | GitHub Actions — lint, mypy, tests | `.github/workflows/ci.yml` |

---

## Project Structure

```
mcp-agents-demo/
├── src/
│   ├── core/
│   │   ├── config.py          # BaseSettings — single source of truth
│   │   ├── cache.py           # Redis cache with in-memory fallback
│   │   ├── concurrency.py     # asyncio · ThreadPool · ProcessPool · Queue
│   │   ├── decorators.py      # @tool · @timed · RateLimit · llm_retry
│   │   ├── logging.py         # structlog JSON structured logging
│   │   ├── models.py          # Pydantic v2 + discriminated unions + generics
│   │   └── tracing.py         # OpenTelemetry — @traced, Jaeger export
│   ├── gateway/
│   │   ├── llm_client.py      # LiteLLM + Helicone unified client
│   │   └── portkey_client.py  # Portkey — guardrails, semantic cache, fallback
│   ├── tools/
│   │   └── shift_tools.py     # Domain tools with @tool decorator
│   ├── agents/
│   │   ├── coverage_agent.py  # Pydantic AI structured-output agent
│   │   ├── langchain_agent.py # LangChain: LCEL · Tool Agent · Streaming
│   │   ├── router_agent.py    # Multi-agent Router → Specialist handoff
│   │   └── runner.py          # All 9 demos — Rich console output
│   └── server/
│       └── main.py            # FastMCP server (tools + resources + prompts)
├── tests/
│   ├── unit/                  # 50 fast tests — no network required
│   └── integration/           # Agent tests in demo mode
├── docs/
│   └── Python_Avanzado_MCP_IA.pdf  # Reference guide
├── .github/workflows/ci.yml
├── litellm_config.yaml        # LiteLLM Proxy config (models, cache, routing)
├── docker-compose.yml         # Full stack: MCP · LiteLLM · Redis · PG · Jaeger
├── Dockerfile
└── pyproject.toml
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
# Set ANTHROPIC_API_KEY for real LLM calls
# Leave empty to run all 9 demos in mock mode (no API key needed)
```

### 3. Run all 9 demos

```bash
uv run demo-agents
```

| Demo | Pattern |
|---|---|
| 1 · Coverage Agent | Pydantic AI structured output |
| 2 · Multi-Agency Parallel | `asyncio.Semaphore` + `gather` |
| 3 · Queue Pipeline | `asyncio.Queue` producer/consumer |
| 4 · ThreadPoolExecutor | Blocking I/O without blocking the event loop |
| 5 · LangChain | LCEL · Tool Agent · Streaming · Parallel chains |
| 6 · Router → Specialist | Multi-agent handoff pattern |
| 7 · Semantic Cache | Redis / in-memory fallback, TTL per namespace |
| 8 · OpenTelemetry | `@traced` decorator, span attributes, Jaeger export |
| 9 · Portkey | Guardrails · semantic cache · multi-provider fallback |

### 4. Run the MCP server

```bash
# stdio (Claude Desktop)
uv run mcp-server

# SSE/HTTP (production)
MCP_TRANSPORT=sse uv run mcp-server
```

### 5. Run tests

```bash
uv run pytest                        # all 50 tests
uv run pytest tests/unit/ -v         # unit tests only
uv run pytest --cov=src --cov-report=html
```

---

## Full Stack with Docker

```bash
docker compose up --build
```

| Service | URL | Purpose |
|---|---|---|
| MCP Server | `http://localhost:8000` | FastMCP SSE endpoint |
| LiteLLM Proxy | `http://localhost:4000` | OpenAI-compatible LLM gateway |
| Redis | `localhost:6379` | Semantic cache |
| Jaeger UI | `http://localhost:16686` | Distributed traces |

---

## Claude Desktop Integration

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "scheduling-demo": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-agents-demo", "mcp-server"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

Available tools: `tool_search_shifts` · `tool_get_schedule` · `tool_update_shift_status` · `tool_analyze_coverage` · `tool_ai_analyze_coverage`

---

## Key Patterns

### Decorator stacking (order matters)
```python
@timed                          # 3rd — measures total time incl. retry
@llm_retry(max_attempts=4)      # 2nd — retries on RateLimitError
@RateLimit(calls_per_second=5)  # 1st — rate limiting closest to function
async def call_llm(prompt: str) -> str: ...
```

### Discriminated Union (O(1) dispatch)
```python
Content = Annotated[
    Union[TextContent, ToolUseContent, ToolResultContent],
    Field(discriminator="type"),
]
```

### Semaphore-limited gather
```python
results = await gather_with_limit(coros, max_concurrent=3)
```

### Multi-agent handoff
```python
router = RouterAgent()
result = await router.run("Analyze coverage risk for agency 42", agency_id=42)
# → RouterAgent classifies → CoverageSpecialist runs → SpecialistResult
```

### Redis cache with fallback
```python
cache = await get_cache()          # Redis if available, else in-memory
hit   = await cache.get("analyze_coverage", payload)
if not hit:
    hit = await analyze_coverage(**payload)
    await cache.set("analyze_coverage", payload, hit)
```

### OpenTelemetry tracing
```python
@traced("llm.call", attributes={"model": "claude-haiku-4-5"})
async def call_llm(prompt: str) -> str: ...
```

---

## Reference Guide

📄 [`docs/Python_Avanzado_MCP_IA.pdf`](docs/Python_Avanzado_MCP_IA.pdf) — guía completa de todos los patrones implementados en este repo.

---

## Roadmap

- [x] FastMCP server — tools, resources, prompt templates
- [x] Pydantic AI — structured-output agent
- [x] LangChain — LCEL · Tool Agent · Streaming · Parallel chains
- [x] Redis-backed semantic cache with in-memory fallback
- [x] Portkey — guardrails, semantic cache, multi-provider fallback
- [x] OpenTelemetry tracing — `@traced` decorator, Jaeger export
- [x] Multi-agent handoff — Router → Specialist pattern
- [x] LiteLLM Proxy — `docker-compose` service with Redis + PostgreSQL
- [ ] CrewAI multi-agent crew demo
- [ ] AutoGen code-execution agent demo

---

## Development

```bash
uv run ruff check src/ tests/   # lint
uv run ruff format src/ tests/  # format
uv run mypy src/                # type check
uv run pre-commit install       # git hooks
```

---

## License

MIT © Erodis Pérez Michel
