# =============================================================================
# ConsensAI API Dockerfile
# =============================================================================
# Uses UV for fast, secure dependency management with lockfile verification.
# See: https://docs.astral.sh/uv/guides/integration/docker/
# =============================================================================

FROM python:3.14-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (better layer caching)
# UV will verify hashes from lockfile for supply chain security
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY configs/ ./configs/

# Install the project itself
RUN uv sync --frozen --no-dev

# Create data directory for SQLite
RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with UV (uses the virtual environment automatically)
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
