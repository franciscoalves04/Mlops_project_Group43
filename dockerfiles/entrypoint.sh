#!/bin/bash
set -e

# Download data from GCS if GCS_DATA_PATH is set
if [ -n "$GCS_DATA_PATH" ]; then
    echo "Downloading data from $GCS_DATA_PATH..."
    mkdir -p /app/data/processed
    python3 << 'PYTHON_EOF'
import os
from google.cloud import storage

gcs_path = os.environ.get("GCS_DATA_PATH")
bucket_name = gcs_path.replace("gs://", "").split("/")[0]
blob_prefix = "/".join(gcs_path.replace("gs://", "").split("/")[1:])

client = storage.Client()
bucket = client.bucket(bucket_name)
blobs = bucket.list_blobs(prefix=blob_prefix)

for blob in blobs:
    if blob.name.endswith("/"):
        continue  # Skip directories
    # Strip the blob_prefix and create file in /app/data/processed/
    relative_path = blob.name.replace(blob_prefix + '/', '')
    local_path = f"/app/data/processed/{relative_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading {blob.name}...")
    blob.download_to_filename(local_path)

print("Data downloaded successfully")
PYTHON_EOF
fi

# Run training
python -m eye_diseases_classification.train "$@"
