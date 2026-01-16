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

# Create directories for data and models with proper permissions
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R mlops:mlops /app/data /app/models /app/logs

USER mlops

# Define volumes for data persistence
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Labels for metadata
LABEL maintainer="Group43" \
      description="Training container for eye diseases classification" \
      version="0.0.1"

ENTRYPOINT ["python", "-m", "eye_deseases_classification.train"]
