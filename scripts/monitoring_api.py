# monitoring_api.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import anyio
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google.cloud import storage

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset


BUCKET_NAME = os.environ["BUCKET_NAME"]  # e.g. group43-eye-monitoring
LOG_PREFIX = os.getenv("LOG_PREFIX", "prediction_logs")
REFERENCE_BLOB = os.getenv("REFERENCE_BLOB", "reference/reference_features.csv")


app = FastAPI()


def _gcs_client() -> storage.Client:
    return storage.Client()


def download_reference(local_path: Path) -> None:
    client = _gcs_client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(REFERENCE_BLOB)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))


def load_latest_prediction_logs(n: int) -> pd.DataFrame:
    client = _gcs_client()
    blobs = list(client.list_blobs(BUCKET_NAME, prefix=LOG_PREFIX))
    if not blobs:
        return pd.DataFrame()

    blobs = sorted(blobs, key=lambda b: b.updated, reverse=True)[:n]

    rows: List[Dict[str, Any]] = []
    for b in blobs:
        try:
            data = json.loads(b.download_as_text())
            feats = data.get("drift_features", {})
            pred = data.get("prediction", {})
            feats["target"] = pred.get("pred_class")  # current "target" = predicted label
            rows.append(feats)
        except Exception:
            continue

    return pd.DataFrame(rows)


def run_evidently(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> None:
    """
    Generates an Evidently HTML report.
    """
    report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
            TargetDriftPreset(columns=["target"]),
        ]
    )
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(out_html))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "bucket": BUCKET_NAME,
        "log_prefix": LOG_PREFIX,
        "reference_blob": REFERENCE_BLOB,
    }


@app.get("/report")
async def report(n: int = 200) -> HTMLResponse:
    """
    Returns an HTML Evidently report for the latest N logs vs reference.
    """
    if n <= 0:
        raise HTTPException(status_code=400, detail="n must be > 0")

    ref_path = Path("ref/reference_features.csv")
    download_reference(ref_path)
    reference = pd.read_csv(ref_path)

    current = load_latest_prediction_logs(n)
    if current.empty:
        raise HTTPException(status_code=404, detail="No prediction logs found in bucket/prefix yet.")

    out_html = Path("monitoring.html")
    run_evidently(reference, current, out_html)

    async with await anyio.open_file(out_html, encoding="utf-8") as f:
        html = await f.read()

    return HTMLResponse(content=html, status_code=200)
