import pytest
import torch
from fastapi.testclient import TestClient
from eye_diseases_classification.api import app

# 1. Define a tiny Dummy Model 
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Batch, 3, 256, 256] -> Output: [Batch, 4] (4 classes)
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x.flatten(1)


@pytest.fixture(name="client")
def client_fixture(tmp_path, monkeypatch):
    """
    Creates a temporary valid ONNX file and forces the API to use it.
    """
    model = DummyModel()
    model.eval()

    dummy_onnx_path = tmp_path / "dummy_model.onnx"

    dummy_input = torch.randn(1, 3, 256, 256)

    torch.onnx.export(
        model,
        dummy_input,
        dummy_onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    monkeypatch.setenv("ONNX_PATH", str(dummy_onnx_path))

    # Initialize the TestClient (this triggers the API startup event)
    with TestClient(app) as client:
        yield client

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Eye Disease Classification Model API!"}

def test_read_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ok"