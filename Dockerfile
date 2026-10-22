# syntax=docker/dockerfile:1
#--------------------------------------------------------------------
# AI Email Assistant — multi-stage production Docker image
#--------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv /opt/venv

# Install runtime dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install the application source.
WORKDIR /app
COPY . .

#--------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PATH="/opt/venv/bin:$PATH"

# Non-root user for running the application.
RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

WORKDIR /app
USER appuser

# Create the data directory for SQLite persistence.
RUN mkdir -p /app/data && chown appuser:appuser /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=3)" || exit 1

ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
