from pathlib import Path
import pandas as pd
from PIL import Image

from eye_diseases_classification.api import extract_drift_features

TRAIN_DIR = Path("data/processed/train")
OUT = Path("reference_features.csv")

rows = []
for cls_dir in TRAIN_DIR.iterdir():
    if not cls_dir.is_dir():
        continue
    label = cls_dir.name
    for p in cls_dir.glob("*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            img = Image.open(p).convert("RGB")
            feats = extract_drift_features(img)
            feats["target"] = label
            rows.append(feats)
        except Exception:
            continue

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Saved {len(df)} rows to {OUT}")
print(df.head())
