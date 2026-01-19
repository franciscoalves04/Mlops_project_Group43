#!/bin/bash
set -e

# Download data from GCS if GCS_DATA_PATH is set
if [ -n "$GCS_DATA_PATH" ]; then
    echo "Downloading data from $GCS_DATA_PATH..."
    mkdir -p /app/data
    gsutil -m cp -r "$GCS_DATA_PATH"/* /app/data/
    echo "Data downloaded successfully"
fi

# Run training
python -m eye_diseases_classification.train "$@"
