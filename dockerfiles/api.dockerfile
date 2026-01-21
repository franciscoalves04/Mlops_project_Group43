# build with "docker build --progress=plain -t eye-api:local -f dockerfiles/api.dockerfile ."
# run with "docker run --rm -it -p 8080:8080 -v "$(pwd)/models:/app/models" eye-api:local"
FROM python:3.12-slim
WORKDIR /app

COPY requirements_api.txt /app/requirements_api.txt
RUN pip install --no-cache-dir -r /app/requirements_api.txt

COPY . /app
ENV PYTHONPATH=/app/src

CMD ["sh", "-c", "uvicorn eye_diseases_classification.api:app --host 0.0.0.0 --port ${PORT:-8080}"]

