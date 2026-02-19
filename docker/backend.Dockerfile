# syntax=docker/dockerfile:1

# ---------- Stage 1: Install dependencies ----------
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency metadata + create minimal src stub so setuptools
# can resolve package-dir and generate egg_info without real source.
COPY pyproject.toml README.md /app/
RUN mkdir -p src/mlforecast_realworld && \
    touch src/mlforecast_realworld/__init__.py

# Install all deps (project itself is a stub; real code comes later)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install .

# ---------- Stage 2: Final image ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (rarely changes → early layer)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Python packages from deps stage (rarely changes)
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Directories for data/artifacts (will be bind-mounted)
RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts/models /app/artifacts/reports

# Entrypoint (rarely changes)
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Project metadata + scripts (change occasionally)
COPY pyproject.toml README.md /app/
COPY scripts /app/scripts

# Source code (changes most often → last layer)
COPY src /app/src

# Install project package only (deps already present)
RUN pip install --no-deps --no-cache-dir .

EXPOSE 8000


ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "mlforecast_realworld.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
