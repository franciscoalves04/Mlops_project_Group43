# run with "uv run uvicorn eye_diseases_classification.api:app --reload --port 8000 --app-dir src"
from __future__ import annotations

import io
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from PIL import Image, UnidentifiedImageError

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
IMAGE_SIZE = (256, 256)

PREDICTION_LOG_BUCKET = os.getenv("PREDICTION_LOG_BUCKET")  # bucket name only (no gs://)
PREDICTION_LOG_PREFIX = os.getenv("PREDICTION_LOG_PREFIX", "prediction_logs")


def normalize_imagenet(x: np.ndarray) -> np.ndarray:
    # x: float32, (C,H,W), in [0,1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (x - mean) / std


def download_from_gcs(gcs_uri: str, local_path: Path) -> None:
    """Download a file from GCS."""
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required for GCS support")

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
            temp_dir = Path(tempfile.gettempdir()) / "model_artifact"
            temp_dir.mkdir(exist_ok=True)

            if gcs_artifact.endswith(".tar.gz"):
                tar_path = temp_dir / "model.tar.gz"
                download_from_gcs(gcs_artifact, tar_path)

                import tarfile

                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(temp_dir)

                onnx_files = list(temp_dir.rglob("*.onnx"))
                if onnx_files:
                    return onnx_files[0]
            else:
                onnx_path = temp_dir / "model.onnx"
                download_from_gcs(gcs_artifact, onnx_path)
                return onnx_path

    explicit = os.getenv("ONNX_PATH")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"ONNX_PATH set but not found: {p}")
        return p

    candidates = sorted(models_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .onnx files found in {models_dir.resolve()}")
    return candidates[0]


def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,C) in [0,1]
    x = np.transpose(arr, (2, 0, 1))  # (C,H,W)
    x = normalize_imagenet(x)
    x = np.expand_dims(x, axis=0).astype(np.float32)  # (1,C,H,W)
    return x


def _to_gray(arr_hwc_01: np.ndarray) -> np.ndarray:
    # arr: (H,W,C) float32 in [0,1]
    r, g, b = arr_hwc_01[..., 0], arr_hwc_01[..., 1], arr_hwc_01[..., 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)


def _laplacian_var(gray_hw: np.ndarray) -> float:
    # Simple 2D Laplacian (no scipy)
    # kernel = [[0,1,0],[1,-4,1],[0,1,0]]
    g = gray_hw
    # pad edge
    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    center = gp[1:-1, 1:-1]
    lap = gp[0:-2, 1:-1] + gp[2:, 1:-1] + gp[1:-1, 0:-2] + gp[1:-1, 2:] - 4.0 * center
    return float(np.var(lap))


def _edge_density(gray_hw: np.ndarray, thresh: float = 0.10) -> float:
    # Edge density via simple gradient magnitude threshold
    g = gray_hw
    gx = np.diff(g, axis=1)
    gy = np.diff(g, axis=0)
    # align shapes
    gx = gx[:-1, :]
    gy = gy[:, :-1]
    mag = np.sqrt(gx * gx + gy * gy)
    return float((mag > thresh).mean())


def extract_drift_features(pil_img: Image.Image) -> Dict[str, float]:
    img = pil_img.convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,C) in [0,1]

    brightness_mean = float(arr.mean())
    brightness_std = float(arr.std())

    gray = _to_gray(arr)
    contrast_std = float(gray.std())
    sharpness_laplacian_var = _laplacian_var(gray)
    edge_density = _edge_density(gray)

    # Optional saturation proxy (cheap): channel std mean
    channel_std_mean = float(arr.std(axis=(0, 1)).mean())

    return {
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "contrast_std": contrast_std,
        "sharpness_laplacian_var": sharpness_laplacian_var,
        "edge_density": edge_density,
        "channel_std_mean": channel_std_mean,
    }


def log_record_to_gcs(record: Dict[str, Any]) -> None:
    """
    Background task: write one JSON record to GCS under a unique key.
    """
    if not (GCS_AVAILABLE and PREDICTION_LOG_BUCKET):
        return  # logging disabled

    try:
        client = storage.Client()
        bucket = client.bucket(PREDICTION_LOG_BUCKET)

        ts = record.get("timestamp_utc", datetime.now(timezone.utc).isoformat())
        day = ts[:10]  # YYYY-MM-DD
        rid = record.get("request_id", str(uuid.uuid4()))
        blob_name = f"{PREDICTION_LOG_PREFIX}/{day}/{ts}_{rid}.json".replace(":", "-")

        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(record, ensure_ascii=False),
            content_type="application/json",
        )
    except Exception as e:
        # Never break inference due to logging
        print(f"[WARN] Failed to log to GCS: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path("models")
    onnx_path = pick_onnx(models_dir)

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
    return {
        "status": "ok",
        "checkpoint": app.state.onnx_path,
        "gcs_logging_enabled": bool(GCS_AVAILABLE and PREDICTION_LOG_BUCKET),
        "log_bucket": PREDICTION_LOG_BUCKET,
        "log_prefix": PREDICTION_LOG_PREFIX,
    }


@app.post("/classify", status_code=HTTPStatus.OK)
async def classify(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported content type: {file.content_type}")

    request_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    try:
        drift_features = extract_drift_features(img)

        x = preprocess(img)  # (1,C,H,W) float32
        outputs = app.state.sess.run(None, {app.state.input_name: x})
        logits = outputs[0]  # (1, num_classes)

        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (probs / probs.sum(axis=1, keepdims=True))[0]  # (num_classes,)

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])

        probs_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        record = {
            "timestamp_utc": timestamp_utc,
            "request_id": request_id,
            "model_path": app.state.onnx_path,
            "endpoint": "/classify",
            "filename": file.filename,
            "content_type": file.content_type,
            "image_size": {"w": IMAGE_SIZE[0], "h": IMAGE_SIZE[1]},
            "prediction": {
                "pred_index": pred_idx,
                "pred_class": pred_class,
                "pred_confidence": pred_conf,
                "probabilities": probs_map,
            },
            "drift_features": drift_features,
        }
        background_tasks.add_task(log_record_to_gcs, record)

        return {
            "pred_index": pred_idx,
            "pred_class": pred_class,
            "probabilities": probs_map,
            "onnx": app.state.onnx_path,
            "filename": file.filename,
            "request_id": request_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
