import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from eye_deseases_classification.model import ResNet
from eye_deseases_classification.data import MyDataset
from pathlib import Path
import csv
from omegaconf import DictConfig


def train(cfg: DictConfig) -> None:
    """Train the ResNet model using provided configuration.

    Args:
        cfg: Hydra configuration object containing training parameters.
    """
    if cfg.experiment.training.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(cfg.experiment.training.device)
    print(f"Using device: {device}")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Description: {cfg.experiment.description}")

    Path(cfg.experiment.checkpoints.save_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.experiment.logging.metrics_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = MyDataset(cfg.experiment.data.train_path)
    val_dataset = MyDataset(cfg.experiment.data.val_path)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.experiment.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.experiment.training.batch_size, shuffle=False
    )

    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.experiment.training.learning_rate)

    csv_file = Path(cfg.experiment.logging.metrics_dir) / cfg.experiment.logging.log_file
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(cfg.experiment.training.epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

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
        print(
            f"Epoch {epoch+1}/{cfg.experiment.training.epochs} | Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2f}%"
        )

        model.eval()
        val_running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_running_loss / len(val_loader)
        val_acc = correct / total * 100
        print(
            f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.2f}%"
        )

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

        if epoch + 1 in cfg.experiment.checkpoints.save_epochs:
            model_path = (
                Path(cfg.experiment.checkpoints.save_dir) / f"resnet_model_epoch{epoch+1}.pt"
            )
            torch.save(model.state_dict(), model_path)
            print(f"Model saved as {model_path}")


if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        train(cfg)

    main()
