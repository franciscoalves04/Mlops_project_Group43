from pathlib import Path
import torch
from torch.utils.data import DataLoader
from eye_deseases_classification.model import ResNet
from eye_deseases_classification.data import MyDataset
import typer

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

def evaluate(model_checkpoint: Path = Path("models/resnet_model.pt"), batch_size: int = 16) -> None:

    print(f"Loading model from {model_checkpoint}...")
    model = ResNet().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()

    # Load test dataset
    test_data_path = Path("data/processed/test")
    test_dataset = MyDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    typer.run(evaluate)
