# build with "docker build --progress=plain -t eye-api:local -f dockerfiles/api.dockerfile ."
# run with "docker run --rm -it -p 80:80 -v "$(pwd)/models:/app/models" eye-api:local"
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app
ENV PYTHONPATH=/app/src
EXPOSE 8080
CMD ["sh", "-c", "uvicorn eye_diseases_classification.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
