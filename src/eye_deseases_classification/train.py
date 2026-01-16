import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from eye_deseases_classification.model import ResNet
from eye_deseases_classification.data import MyDataset
from pathlib import Path
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 70
LEARNING_RATE = 1e-3

# Paths
MODEL_DIR = Path("models")
DOCS_DIR = Path("docs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def train():
    # Load datasets
    train_dataset = MyDataset("data/processed/train")
    val_dataset = MyDataset("data/processed/val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, loss, optimizer
    model = ResNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CSV file - metrics
    csv_file = DOCS_DIR / "training_metrics.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_acc = correct / total * 100
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        # Save metrics to CSV
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

        # Save model at epoch 50, 60, 70
        if epoch + 1 in [50, 60, 70]:
            model_path = MODEL_DIR / f"resnet_model_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved as {model_path}")

if __name__ == "__main__":
    train()
