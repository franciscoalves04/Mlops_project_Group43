#!/bin/bash
set -e

# Download data from GCS if GCS_DATA_PATH is set
if [ -n "$GCS_DATA_PATH" ]; then
    echo "Downloading data from $GCS_DATA_PATH..."
    mkdir -p /app/data/processed
    python3 << 'PYTHON_EOF'
import os
import sys
from google.cloud import storage

try:
    gcs_path = os.environ.get("GCS_DATA_PATH")
    if not gcs_path:
        print("ERROR: GCS_DATA_PATH not set")
        sys.exit(1)
    
    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    blob_prefix = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    
    print(f"Bucket: {bucket_name}, Prefix: {blob_prefix}")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=blob_prefix))
    
    if not blobs:
        print(f"ERROR: No blobs found in {gcs_path}")
        sys.exit(1)
    
    print(f"Found {len(blobs)} blobs to download")
    
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
except Exception as e:
    print(f"ERROR during GCS download: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF
    if [ $? -ne 0 ]; then
        echo "ERROR: Data download failed!"
        exit 1
    fi
else
    echo "WARNING: GCS_DATA_PATH not set, skipping data download"
fi

# Run training
python -m eye_diseases_classification.train "$@"
