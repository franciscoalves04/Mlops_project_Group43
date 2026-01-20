import os
import pytest
import torch

from eye_diseases_classification.data import MyDataset, normalize_image

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/processed/test")


@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason="Data folder not found")
def test_dataset_loading():
    dataset = MyDataset(DATA_PATH)

    # Dataset should not be empty
    assert len(dataset) > 0, "Dataset is empty"

    # Check one sample
    img, label = dataset[0]

    assert isinstance(img, torch.Tensor), "Image is not a torch.Tensor"
    assert img.shape == (3, 256, 256), f"Unexpected image shape: {img.shape}"
    assert isinstance(label, int), "Label is not an integer"
    assert 0 <= label < 4, f"Label {label} out of range"

    # Check that all labels are represented
    labels = [dataset[i][1] for i in range(len(dataset))]
    for label in range(4):
        assert label in labels, f"Label {label} missing from dataset"


def test_normalize_image_shape():
    img = torch.rand(3, 256, 256)
    out = normalize_image(img)

    assert out.shape == img.shape, "Normalization changed tensor shape"
