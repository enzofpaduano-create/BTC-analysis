# Multi-stage build for the alerts runner.
# Image is intentionally Python 3.12-slim — same version as our pinned Python
# in `.python-version`, no compiler bloat needed at runtime since we ship wheels.

FROM python:3.12-slim AS base

# Install uv globally (small, fast). One-line installer; no curl trickery.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# System libs needed by hmmlearn / arch / pyarrow at runtime (already shipped
# as wheels but a couple of native libs are helpful for hmmlearn).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first so dependency layers are cached separately
# from source code changes.
COPY pyproject.toml uv.lock .python-version ./

# Install the data + features + backtest extras only — the alerts runner
# needs everything except the dev tools.
RUN uv sync --frozen --no-dev --extra data --extra features --extra backtest

# Now copy the project source.
COPY core ./core
COPY data ./data
COPY features ./features
COPY signals ./signals
COPY backtest ./backtest
COPY live ./live
COPY scripts ./scripts

# Mountable directory for the Parquet store + DuckDB + alerts.jsonl.
# Cloud providers should mount a volume here so state survives restarts.
RUN mkdir -p /app/data_store /app/reports /app/logs
VOLUME ["/app/data_store", "/app/reports"]

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data_store

# A non-root user is good hygiene.
RUN useradd --uid 10001 --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "--no-sync", "python", "-m", "scripts.run_alerts"]
