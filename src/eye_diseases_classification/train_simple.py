"""Simple training script without Hydra config - for backwards compatibility."""
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from eye_diseases_classification.data import MyDataset
from eye_diseases_classification.model import ResNet


def main():
    """Simple training without config files."""
    # Load datasets with augmentation for training only
    train_dataset = MyDataset("data/processed/train", augment=True)
    val_dataset = MyDataset("data/processed/val", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="models",
        filename="best_model-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Logger
    logger = CSVLogger(save_dir="logs", name="simple_training")

    # Trainer
    trainer = Trainer(
        max_epochs=30,
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=logger,
        deterministic=True,
        log_every_n_steps=None,
    )

    # Model
    model = ResNet(num_classes=4, lr=0.001)

    # Train
    print("Starting simple training (no Hydra config)...")
    trainer.fit(model, train_loader, val_loader)
    print(f"Training complete! Best model: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
