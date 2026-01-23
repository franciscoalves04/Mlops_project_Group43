FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements_frontend.txt /app/requirements_frontend.txt
RUN pip install --no-cache-dir -r /app/requirements_frontend.txt

COPY src/eye_diseases_classification/api_frontend.py /app/frontend.py

EXPOSE 8080

CMD ["sh", "-c", "streamlit run frontend.py --server.address=0.0.0.0 --server.port ${PORT:-8080} --server.headless true"]
