from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F

@dataclass
class OnnxInputSpec:
    name: str
    c: int
    h: int
    w: int


def find_latest_onnx_model(models_dir: str = "models") -> str:
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    onnx_files = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".onnx")
    ]

    if not onnx_files:
        raise FileNotFoundError(f"No .onnx models found in {models_dir}")

    latest = max(onnx_files, key=os.path.getmtime)
    print(f"Using latest ONNX model: {latest}")
    return latest


def infer_onnx_input_spec(onnx_path: str) -> OnnxInputSpec:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = inp.shape  # expected [N,C,H,W]

    if len(shape) != 4:
        raise ValueError(f"Expected NCHW ONNX input, got shape={shape}")

    _, c, h, w = shape
    if not all(isinstance(x, int) for x in [c, h, w]):
        raise ValueError(f"ONNX input must have fixed C,H,W. Got {shape}")

    return OnnxInputSpec(name=inp.name, c=c, h=h, w=w)

def get_test_loader(spec: OnnxInputSpec) -> DataLoader:
    from torchvision.datasets import ImageFolder

    test_dir = "data/processed/test"
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Expected test set at {test_dir}/<class_name>/*.jpg"
        )

    transform = T.Compose(
        [
            T.Resize((spec.h, spec.w)),
            T.ToTensor(),  # float32 [0,1], [C,H,W]
        ]
    )

    ds = ImageFolder(test_dir, transform=transform)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

def build_scenarios():
    def brightness(delta):
        return lambda x: F.adjust_brightness(x, 1.0 + delta)

    def contrast(delta):
        return lambda x: F.adjust_contrast(x, 1.0 + delta)

    def blur(k, s):
        g = T.GaussianBlur(kernel_size=k, sigma=s)
        return lambda x: g(x)

    def noise(std):
        def _t(x):
            return torch.clamp(x + torch.randn_like(x) * std, 0, 1)
        return _t

    return [
        ("baseline", lambda x: x),
        ("brightness+0.2", brightness(0.2)),
        ("brightness-0.2", brightness(-0.2)),
        ("contrast+0.3", contrast(0.3)),
        ("contrast-0.3", contrast(-0.3)),
        ("blur_k3_s1.0", blur(3, 1.0)),
        ("blur_k5_s2.0", blur(5, 2.0)),
        ("noise_std0.02", noise(0.02)),
        ("noise_std0.05", noise(0.05)),
    ]


def make_preprocess(scenario_fn, spec: OnnxInputSpec):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preprocess(batch):
        xs = []
        for i in range(batch.shape[0]):
            xs.append(scenario_fn(batch[i]))
        x = torch.stack(xs)

        if spec.c == 1 and x.shape[1] == 3:
            x = (
                0.2989 * x[:, 0:1]
                + 0.5870 * x[:, 1:2]
                + 0.1140 * x[:, 2:3]
            )
        elif spec.c == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if spec.c == 3:
            x = (x - mean) / std

        return x.float()

    return preprocess

def load_model(onnx_path: str, spec: OnnxInputSpec) -> torch.nn.Module:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = spec.name

    class ONNXModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_np = x.detach().cpu().numpy().astype(np.float32)
            out = sess.run(None, {input_name: x_np})[0]
            return torch.from_numpy(out)

    return ONNXModel()

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


@torch.inference_mode()
def evaluate(model, loader, preprocess):
    accs = []
    for x, y in loader:
        x = preprocess(x)
        logits = model(x)
        accs.append(accuracy(logits, y))
    return sum(accs) / len(accs)

def main():
    onnx_path = find_latest_onnx_model("models")
    spec = infer_onnx_input_spec(onnx_path)

    print(f"ONNX input: [N,{spec.c},{spec.h},{spec.w}]")

    loader = get_test_loader(spec)
    model = load_model(onnx_path, spec)

    results = []
    for name, tfm in build_scenarios():
        preprocess = make_preprocess(tfm, spec)
        acc = evaluate(model, loader, preprocess)
        results.append((name, acc))

    print("\nRobustness to synthetic drift (accuracy):")
    for n, a in results:
        print(f"{n:20s} acc={a:.4f}")

    baseline = dict(results)["baseline"]
    print("\nDrop vs baseline:")
    for n, a in results:
        if n != "baseline":
            print(f"{n:20s} drop={baseline - a:.4f}")


if __name__ == "__main__":
    main()
