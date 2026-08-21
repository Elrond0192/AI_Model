# ─── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install build tools required by scientific Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency manifests first for layer caching
COPY pyproject.toml requirements.txt README.md ./
COPY basketball_ai/__init__.py ./basketball_ai/__init__.py

# Install dependencies into a prefix directory
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install the project package itself
COPY . .
RUN pip install --no-cache-dir --prefix=/install --no-deps .

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy only the installed packages and project files from the builder
COPY --from=builder /install /usr/local
COPY --from=builder /app /app

# Default data/model directories (can be overridden at runtime via env vars)
ENV DATA_DIR=/app/data/sample \
    MODEL_DIR=/app/models_saved \
    DATA_SOURCE=postgres \
    LOG_LEVEL=INFO \
    API_ENV=production \
    ALLOWED_ORIGINS="" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

# Create directories expected by the app
RUN mkdir -p /app/data/sample /app/models_saved \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 8501

# The API service uses this image-level healthcheck. The admin service overrides
# it in docker-compose.yml with /healthz on port 8501.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

CMD ["uvicorn", "basketball_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
