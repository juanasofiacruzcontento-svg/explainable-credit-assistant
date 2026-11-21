FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp serve a XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"

COPY app ./app
COPY knowledge ./knowledge
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000 8501
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "credit_xai.api:app", "--host", "0.0.0.0", "--port", "8000"]
