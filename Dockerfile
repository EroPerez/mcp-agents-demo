FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── deps layer (cached unless pyproject.toml changes) ────────────────────────
FROM base AS deps
RUN pip install uv
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# ── runtime ───────────────────────────────────────────────────────────────────
FROM base AS runtime
COPY --from=deps /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY src/ ./src/

# MCP server (SSE transport for HTTP)
ENV MCP_TRANSPORT=sse
EXPOSE 8000
CMD ["python", "-m", "src.server.main"]
