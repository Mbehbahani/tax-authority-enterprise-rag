# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — compile native wheels
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY docker/requirements.txt ./requirements.txt

RUN pip install --upgrade pip wheel && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — minimal image, non-root user, tini PID 1
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# tini as PID 1 for signal propagation and zombie reaping
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r app && useradd -r -g app -m -d /home/app app

WORKDIR /app

# Install wheels from builder
COPY --from=builder /build/wheels /wheels
COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt \
    && rm -rf /wheels

# Copy application source
COPY tests/ /app/tests/
COPY tests/golden/ /app/tests/golden/

# Create directories for reports
RUN mkdir -p /app/reports && chown -R app:app /app

# Switch to non-root
USER app

# AWS credential env vars are NOT baked in — they are injected at runtime via .env
# These are documented in .env.example

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Startup probe: STS GetCallerIdentity — logs IAM ARN before serving
# This is run by the healthcheck in compose; the app entrypoint is pytest
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short", \
     "--junitxml=/app/reports/junit.xml", \
     "-m", "not slow"]
