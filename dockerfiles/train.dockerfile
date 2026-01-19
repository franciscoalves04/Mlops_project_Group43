FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies without the project itself
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY src ./src

# Install the project
RUN uv sync --frozen --no-dev

# Runtime stage
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install uv in runtime image
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies in runtime image (not builder's venv)
RUN uv sync --frozen --no-dev

# Copy source code
COPY src ./src

# Copy entrypoint script
COPY dockerfiles/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

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

ENTRYPOINT ["/app/entrypoint.sh"]
