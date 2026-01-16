FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies without the project itself
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY src ./src

# Install the project
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy uv and virtual environment from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd -m -u 1000 mlops && \
    chown -R mlops:mlops /app

# Create directory for models with proper permissions
RUN mkdir -p /app/models && \
    chown -R mlops:mlops /app/models

USER mlops

# Expose API port
EXPOSE 8000

# Define volume for model persistence
VOLUME ["/app/models"]

# Labels for metadata
LABEL maintainer="Group43" \
      description="API inference container for eye diseases classification" \
      version="0.0.1"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "eye_deseases_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]
