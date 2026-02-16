FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Create directories for data and artifacts (will be mounted as volumes)
RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts/models /app/artifacts/reports

EXPOSE 8000

# Use an entrypoint script that can optionally pre-download data
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "mlforecast_realworld.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
