FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_api.txt /app/requirements_api.txt


RUN pip install --no-cache-dir -r /app/requirements_api.txt

COPY models /app/models

COPY . /app
ENV PYTHONPATH=/app/src

CMD ["sh", "-c", "uvicorn eye_diseases_classification.api:app --host 0.0.0.0 --port ${PORT:-8080}"]