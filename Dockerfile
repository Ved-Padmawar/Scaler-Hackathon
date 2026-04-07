FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast, openenv standard)
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY env/ ./env/
COPY graders/ ./graders/
COPY server/ ./server/
COPY data/ ./data/
COPY inference.py ./
COPY run.py ./
COPY openenv.yaml ./

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run via venv python directly (avoids uv rebuild of project at runtime)
CMD [".venv/bin/python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
