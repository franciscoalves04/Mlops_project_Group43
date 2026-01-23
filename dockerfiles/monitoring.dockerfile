FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_monitoring.txt /app/requirements_monitoring.txt
RUN pip install --no-cache-dir -r /app/requirements_monitoring.txt

COPY scripts/monitoring_api.py /app/monitoring_api.py


ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn monitoring_api:app --host 0.0.0.0 --port $PORT --workers 1
