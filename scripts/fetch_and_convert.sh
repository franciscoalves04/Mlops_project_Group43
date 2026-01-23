#!/bin/bash
set -e

# Configuration
PROJECT_ID=$1
REGION=$2
# Your teammate's script hardcodes this directory, so we must use it
MODEL_DIR="models"

echo "--- 1. Finding Best Accuracy Model ---"

RAW_VERSIONS=$(gcloud artifacts versions list \
    --package="eye-diseases-model" \
    --repository="mlops-models" \
    --location=$REGION \
    --project=$PROJECT_ID \
    --format="value(name)")

# Python script to pick the winner based on the "acc" tag
BEST_VERSION=$(python3 -c "
import sys
import re

lines = sys.stdin.read().splitlines()
best_ver = None
max_acc = -1.0
pattern = re.compile(r'acc(\d+\.\d+)$')

for line in lines:
    match = pattern.search(line)
    if match:
        acc = float(match.group(1))
        if acc > max_acc:
            max_acc = acc
            best_ver = line

if best_ver:
    print(best_ver)
else:
    sys.exit(1)
" <<< "$RAW_VERSIONS")

if [ -z "$BEST_VERSION" ]; then
    echo "Error: Could not find any models with 'acc' tag!"
    exit 1
fi

VERSION_ID=$(basename "$BEST_VERSION")
ACC_VAL=$(echo "$VERSION_ID" | grep -oP 'acc\K[\d\.]+')

echo "🏆 FOUND CHAMPION MODEL!"
echo "Version:  $VERSION_ID"
echo "Accuracy: $ACC_VAL"

echo "--- 2. Downloading Artifact ---"
mkdir -p download_temp
gcloud artifacts generic download \
    --package="eye-diseases-model" \
    --repository="mlops-models" \
    --location=$REGION \
    --project=$PROJECT_ID \
    --version=$VERSION_ID \
    --destination=download_temp

echo "--- 3. Preparing Checkpoint ---"
# Extract the tarball
tar -xzf "download_temp/model.tar.gz" -C download_temp

# Find the .ckpt file (The teammate's script needs .ckpt, NOT .pth)
CKPT_FILE=$(find download_temp -name "*.ckpt" | head -n 1)

if [ -z "$CKPT_FILE" ]; then
    echo "Error: .ckpt file not found in artifact!"
    exit 1
fi
echo "Found Checkpoint: $CKPT_FILE"

# Prepare the 'models' directory exactly as the script expects
mkdir -p $MODEL_DIR
# Clean old files to ensure the script picks the new one
rm -f $MODEL_DIR/*.ckpt $MODEL_DIR/*.onnx

# Move the checkpoint to 'models/'
cp "$CKPT_FILE" "$MODEL_DIR/champion.ckpt"

echo "--- 4. Running Export Script ---"
# We assume export_onnx.py is in the root or accessible. 
# If it's in 'src/', change to: uv run python src/export_onnx.py
# setting PYTHONPATH=. ensures it can find 'eye_diseases_classification'
PYTHONPATH=. uv run python src/eye_diseases_classification/export_onnx.py

echo "--- Cleanup ---"
rm -rf download_temp

echo "--- Verification ---"
if ls $MODEL_DIR/*.onnx 1> /dev/null 2>&1; then
    echo "✅ Success! ONNX model generated in $MODEL_DIR/"
else
    echo "❌ Error: ONNX file was not generated."
    exit 1
fi