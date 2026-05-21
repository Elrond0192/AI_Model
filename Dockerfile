# ─── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install build tools and ODBC driver dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency manifests first for layer caching
COPY pyproject.toml README.md ./
COPY basketball_ai/__init__.py ./basketball_ai/__init__.py

# Install dependencies into a prefix directory
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install \
        "uvicorn[standard]>=0.29" \
        "fastapi>=0.111" \
        "xgboost>=2.0" \
        "scikit-learn>=1.4" \
        "pandas>=2.0" \
        "numpy>=1.26" \
        "joblib>=1.3" \
        "python-dotenv>=1.0" \
        "slowapi>=0.1.9" \
        "PyJWT>=2.8" \
        "cachetools>=5.3"

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
    DATA_SOURCE=file \
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

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

CMD ["uvicorn", "basketball_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
