FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SUPPORTFLOW_DATA_DIR=/data

RUN useradd --create-home --uid 10001 supportflow

COPY supportflow /app/supportflow

RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    "uvicorn[standard]==0.34.2" \
    langgraph==0.2.19 \
    "celery[redis]==5.4.0" \
    "psycopg[binary]>=3.2,<4" \
    python-dotenv==1.0.1 \
    openai==2.53.0 \
    && mkdir -p /data \
    && chown -R supportflow:supportflow /app /data

USER supportflow

EXPOSE 8000

# Render injects PORT at runtime; local Docker continues to use 8000 when it is absent.
CMD ["sh", "-c", "python -m uvicorn supportflow.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
