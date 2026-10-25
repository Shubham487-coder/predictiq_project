# ============================================================
# PredictIQ — FastAPI Inference Service
# ============================================================
# Multi-stage build: build deps in builder, copy to slim runtime.
# Final image size ~1.5 GB (PyTorch + XGBoost unavoidable).
# ============================================================

FROM python:3.11-slim AS builder

# Build dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -g 1000 predictiq && useradd -u 1000 -g predictiq -m -s /bin/bash predictiq

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source (models + api code)
COPY api/            ./api/
COPY ml_models/      ./ml_models/

# Ensure data/processed directory exists for airflow task output
RUN mkdir -p /app/data/processed && chown -R predictiq:predictiq /app

USER predictiq

# Health check — waits up to 2 minutes for models to load
HEALTHCHECK --interval=15s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -sf http://localhost:8000/api/v1/health | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data',{}).get('models_loaded') else 1)" \
    || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--no-access-log"]