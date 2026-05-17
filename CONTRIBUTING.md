# Contributing

## Setup

```bash
git clone https://github.com/EroPerez/mcp-agents-demo
cd mcp-agents-demo
uv sync
cp .env.example .env
pre-commit install
```

## Workflow

1. Branch from `develop`: `git checkout -b feat/your-feature`
2. Commit using Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
3. Run `uv run ruff check . && uv run pytest` before pushing
4. Open a PR against `develop`

## Code conventions

- **Type annotations** on all public functions
- **Docstrings** on all public classes and functions (Google style)
- **No `time.sleep()`** inside coroutines — use `await asyncio.sleep()`
- **Pydantic models** for all data crossing API boundaries
- **structlog** for all logging (no `print()` in production code)
