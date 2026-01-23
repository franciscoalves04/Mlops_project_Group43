from __future__ import annotations

import io
import os
import tempfile
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
IMAGE_SIZE = (256, 256)

def normalize_imagenet(x: np.ndarray) -> np.ndarray:
    # x: float32, (C,H,W), in [0,1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (x - mean) / std

def download_from_gcs(gcs_uri: str, local_path: Path) -> None:
    """Download a file from GCS."""
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required for GCS support")
    
    # Parse gs://bucket/path/to/file
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    print(f"Downloaded {gcs_uri} to {local_path}")

def pick_onnx(models_dir: Path) -> Path:
    # Check for GCS artifact path first
    gcs_artifact = os.getenv("GCS_MODEL_ARTIFACT")
    if gcs_artifact:
        print(f"Loading model from GCS artifact: {gcs_artifact}")
        if GCS_AVAILABLE:
            # Download from GCS
            temp_dir = Path(tempfile.gettempdir()) / "model_artifact"
            temp_dir.mkdir(exist_ok=True)
            
            # If it's a tar.gz, download and extract
            if gcs_artifact.endswith(".tar.gz"):
                tar_path = temp_dir / "model.tar.gz"
                download_from_gcs(gcs_artifact, tar_path)
                
                import tarfile
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
                
                # Find ONNX file in extracted directory
                onnx_files = list(temp_dir.rglob("*.onnx"))
                if onnx_files:
                    return onnx_files[0]
            else:
                # Direct ONNX file
                onnx_path = temp_dir / "model.onnx"
                download_from_gcs(gcs_artifact, onnx_path)
                return onnx_path
    
    # Prefer explicit env var if set (local path)
    explicit = os.getenv("ONNX_PATH")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"ONNX_PATH set but not found: {p}")
        return p

    # Fall back to local models directory
    candidates = sorted(models_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .onnx files found in {models_dir.resolve()}")
    return candidates[0]

def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H,W,C)
    x = np.transpose(arr, (2, 0, 1))                         # (C,H,W)
    x = normalize_imagenet(x)
    x = np.expand_dims(x, axis=0).astype(np.float32)         # (1,C,H,W)
    return x

@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path("models")
    onnx_path = pick_onnx(models_dir)

    # CPU execution provider (Cloud Run CPU)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    app.state.sess = sess
    app.state.input_name = input_name
    app.state.onnx_path = str(onnx_path)

    yield

    del app.state.sess

app = FastAPI(lifespan=lifespan)

@app.get("/", status_code=HTTPStatus.OK)
def root() -> Dict[str, Any]:
    return {"message": "Welcome to the Eye Disease Classification Model API!"}

@app.get("/health", status_code=HTTPStatus.OK)
def health() -> Dict[str, Any]:
    return {"status": "ok", "checkpoint": app.state.onnx_path}

@app.post("/classify", status_code=HTTPStatus.OK)
async def classify(file: UploadFile = File(...)) -> Dict[str, Any]:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        x = preprocess(img)  # (1,C,H,W) float32

        outputs = app.state.sess.run(None, {app.state.input_name: x})
        logits = outputs[0]  # shape (1, num_classes)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (probs / probs.sum(axis=1, keepdims=True))[0]  # (num_classes,)

        pred_idx = int(np.argmax(probs))
        probs_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        return {
            "pred_index": pred_idx,
            "pred_class": CLASS_NAMES[pred_idx],
            "probabilities": probs_map,
            "onnx": app.state.onnx_path,
            "filename": file.filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
