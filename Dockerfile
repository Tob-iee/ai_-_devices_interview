# Base image
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    TRANSFORMERS_CACHE=/models/hf \
    HF_HOME=/models/hf

# System deps you’ll likely need
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential tini && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
ARG POETRY_VERSION=1.8.3
RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Copy only dependency files for better layer caching
COPY pyproject.toml poetry.lock ./

# Install deps (no dev) into the global env
RUN poetry install --no-dev --no-ansi


# Final image — copy deps from base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/models/hf \
    HF_HOME=/models/hf \
    PORT=8080

# tini for proper signal handling
RUN apt-get update && apt-get install -y --no-install-recommends tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring in the installed site-packages and poetry bins from the base layer
COPY --from=base /usr/local /usr/local

# App source
COPY src/ ./src/

# Expose for local dev; Render injects $PORT at runtime
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini","--"]
# Use shell form so $PORT expands on Render
CMD bash -lc 'streamlit run src/app.py --server.address 0.0.0.0 --server.port ${PORT:-8080}'
