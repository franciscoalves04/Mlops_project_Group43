from __future__ import annotations

import io
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from eye_diseases_classification.model import ResNet
from eye_diseases_classification.data import normalize_image, IMAGE_SIZE


CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]


def pick_checkpoint(models_dir: Path) -> Path:
    ckpts = sorted(models_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {models_dir.resolve()}")
    return ckpts[0]


def preprocess_image_like_dataset(pil_img: Image.Image) -> torch.Tensor:
    """
    Matches MyDataset.__getitem__ when augment=False and transform=None.
    Output: torch.Tensor (C,H,W), float32, normalized.
    """
    img = pil_img.convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0   # (H,W,C) in [0,1]
    x = torch.from_numpy(arr).permute(2, 0, 1)      # (C,H,W)

    x = normalize_image(x)                          # ImageNet normalize
    return x


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path("models")
    checkpoint_path = pick_checkpoint(models_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNet.load_from_checkpoint(str(checkpoint_path))
    model.eval()
    model.to(device)

    app.state.model = model
    app.state.device = device
    app.state.checkpoint_path = str(checkpoint_path)

    yield

    # cleanup
    del app.state.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.get("/health", status_code=HTTPStatus.OK)
def health() -> Dict[str, Any]:
    return {"status": "ok", "device": app.state.device, "checkpoint": app.state.checkpoint_path}


@app.get("/", status_code=HTTPStatus.OK)
def root() -> Dict[str, Any]:
    return {"message": "Welcome to the Eye Disease Classification Model API!"}


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
        x = preprocess_image_like_dataset(img)          # (C,H,W)
        x = x.unsqueeze(0).to(app.state.device)         # (1,C,H,W)

        with torch.inference_mode():
            logits = app.state.model(x)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if isinstance(logits, dict):
                logits = logits.get("logits", None) or logits.get("preds", None)
                if logits is None:
                    raise RuntimeError("Model returned dict without 'logits'/'preds'.")

            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

        pred_idx = int(torch.argmax(probs).item())
        probs_map = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))}

        return {
            "pred_index": pred_idx,
            "pred_class": CLASS_NAMES[pred_idx],
            "probabilities": probs_map,
            "device": app.state.device,
            "checkpoint": app.state.checkpoint_path,
            "filename": file.filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
