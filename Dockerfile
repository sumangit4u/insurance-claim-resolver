# Multi-stage Dockerfile for Cloud Run deployment
# Week 10 deliverable — production-hardened image

# --------------------------------------------------------------------------
# Stage 1: builder
# --------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# --------------------------------------------------------------------------
# Stage 2: runtime
# --------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Non-root user for Cloud Run security best practice
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY agent/ ./agent/
COPY api/ ./api/
COPY config/ ./config/
COPY evaluation/ ./evaluation/
COPY mcp_server/ ./mcp_server/
COPY observability/ ./observability/
COPY prompts/ ./prompts/
COPY rag/ ./rag/
COPY workflow/ ./workflow/

# Copy data directory (policies/SOPs only — claims data in Firestore in prod)
COPY data/policies/ ./data/policies/
COPY data/sops/ ./data/sops/
COPY data/api_spec/ ./data/api_spec/

# Switch to non-root user
USER appuser

# Cloud Run injects PORT env var; default 8000 for local
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV ENVIRONMENT=production

EXPOSE 8000

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2"]
