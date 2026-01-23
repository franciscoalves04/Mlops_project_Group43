from __future__ import annotations

from pathlib import Path
import torch

from eye_diseases_classification.model import ResNet
from eye_diseases_classification.data import IMAGE_SIZE


def pick_latest_checkpoint(models_dir: Path) -> Path:
    ckpts = sorted(models_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {models_dir.resolve()}")
    return ckpts[0]


def main() -> None:
    models_dir = Path("models")
    ckpt_path = pick_latest_checkpoint(models_dir)

    onnx_path = ckpt_path.with_suffix(".onnx")

    device = "cpu"
    model = ResNet.load_from_checkpoint(str(ckpt_path))
    model.eval().to(device)

    dummy = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"Exported model: {onnx_path.name}")


if __name__ == "__main__":
    main()
