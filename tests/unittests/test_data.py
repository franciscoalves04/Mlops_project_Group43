import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from eye_diseases_classification.data import MyDataset, normalize_image, augment_image

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/test")


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Data folder not found")
def test_dataset_loading():
    dataset = MyDataset(DATA_PATH)

    assert len(dataset) > 0, "Dataset is empty"

    img, label = dataset[0]

    assert isinstance(img, torch.Tensor), "Image is not a torch.Tensor"
    assert img.shape == (3, 256, 256), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, int), "Label is not an integer"
    assert 0 <= label < 4, f"Label {label} out of range"

    labels = [dataset[i][1] for i in range(len(dataset))]
    for label in range(4):
        assert label in labels, f"Label {label} missing from dataset"


def test_normalize_image_shape():
    img = torch.rand(3, 256, 256)
    out = normalize_image(img)

    assert out.shape == img.shape, "Normalization changed tensor shape"


def _make_image(path: Path, size=(32, 32), color=(128, 128, 128)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path)


def test_label_mapping_is_deterministic(tmp_path: Path):
    # Create two class folders with one image each; labels should follow sorted order: 'alpha' -> 0, 'zeta' -> 1
    alpha_img = tmp_path / "alpha" / "a.png"
    zeta_img = tmp_path / "zeta" / "z.png"
    _make_image(alpha_img)
    _make_image(zeta_img)

    dataset = MyDataset(tmp_path)
    assert len(dataset) == 2

    # Verify labels from internal samples (stable mapping by sorted class names)
    labels_by_class = {p.parent.name: lbl for p, lbl in dataset.samples}
    assert labels_by_class["alpha"] == 0
    assert labels_by_class["zeta"] == 1


def test_ignores_non_image_files(tmp_path: Path):
    # Create one valid image and one non-image file in the same class
    valid_img = tmp_path / "classA" / "img.png"
    _make_image(valid_img)
    non_img = tmp_path / "classA" / "notes.txt"
    non_img.write_text("not an image")

    dataset = MyDataset(tmp_path)
    assert len(dataset) == 1


def test_transform_applied_before_normalization(tmp_path: Path):
    # Create a single image and use a transform that zeros the tensor; normalized output should equal (-mean)/std
    img_path = tmp_path / "classA" / "img.png"
    _make_image(img_path)

    def zero_transform(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    dataset = MyDataset(tmp_path, transform=zero_transform, augment=False)
    out, label = dataset[0]
    assert out.shape == (3, 256, 256)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    expected = (torch.zeros_like(out) - mean) / std
    assert torch.allclose(out, expected, atol=1e-5)


def test_preprocess_creates_expected_splits_and_sizes(tmp_path: Path):
    # Build small dataset: classA with 10 images, classB with 5 images
    for i in range(10):
        _make_image(tmp_path / "classA" / f"a_{i}.png")
    for i in range(5):
        _make_image(tmp_path / "classB" / f"b_{i}.png")

    dataset = MyDataset(tmp_path)
    out_dir = tmp_path / "processed"
    dataset.preprocess(out_dir)

    # Expected counts based on SPLIT_RATIOS with floor rounding
    # classA: 10 -> train=7, val=1, test=2
    # classB: 5  -> train=3, val=0, test=2
    def count_images(p: Path) -> int:
        return len([f for f in p.iterdir() if f.is_file()]) if p.exists() else 0

    assert count_images(out_dir / "train" / "classA") == 7
    assert count_images(out_dir / "val" / "classA") == 1
    assert count_images(out_dir / "test" / "classA") == 2

    assert count_images(out_dir / "train" / "classB") == 3
    assert count_images(out_dir / "val" / "classB") == 0
    assert count_images(out_dir / "test" / "classB") == 2

    # Verify images were resized to 256x256
    sample_out = next((out_dir / "train" / "classA").iterdir())
    with Image.open(sample_out) as im:
        assert im.size == (256, 256)


def test_augment_image_all_branches(monkeypatch):
    # Sequence to trigger: hflip, vflip, rot, brightness on, brightness factor, contrast on, contrast factor
    seq = iter([0.6, 0.6, 0.9, 0.6, 0.7, 0.6, 0.7])

    def fake_rand(*args, **kwargs) -> torch.Tensor:
        return torch.tensor([next(seq)])

    monkeypatch.setattr(torch, "rand", fake_rand)
    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([2]))

    x = torch.ones(3, 64, 64)
    y = augment_image(x)
    assert y.shape == x.shape
    assert torch.min(y) >= 0 and torch.max(y) <= 1


def test_dataset_augmentation_applies(tmp_path: Path, monkeypatch):
    img_path = tmp_path / "classA" / "img.png"
    _make_image(img_path)

    # Force augmentations to run
    def always_one(_: int) -> torch.Tensor:
        return torch.tensor([0.9])

    monkeypatch.setattr(torch, "rand", always_one)
    monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([1]))

    ds = MyDataset(tmp_path, augment=True)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 256, 256)
    assert isinstance(label, int)


def test_cli_preprocess_runs(tmp_path: Path, monkeypatch):
    # Prepare minimal dataset and run module as CLI
    for i in range(2):
        _make_image(tmp_path / "alpha" / f"a_{i}.png")
    out_dir = tmp_path / "out"

    import runpy
    import sys

    old_argv = sys.argv[:]
    sys.argv = ["data", str(tmp_path), str(out_dir)]
    try:
        import pytest as _pytest

        with _pytest.raises(SystemExit) as e:
            runpy.run_module("eye_diseases_classification.data", run_name="__main__")
        assert e.value.code == 0
    finally:
        sys.argv = old_argv

    assert (out_dir / "train" / "alpha").exists()
