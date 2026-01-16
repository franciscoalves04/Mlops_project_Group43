from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from eye_deseases_classification.data import MyDataset
from eye_deseases_classification.model import ResNet  # LightningModule já configurado


def main():
    # Load datasets
    train_dataset = MyDataset("data/processed/train")
    val_dataset = MyDataset("data/processed/val")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="models",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    # Logger CSV - garante que só grava uma linha por epoch
    logger = CSVLogger(save_dir="logs", name="resnet_training")

    # Trainer
    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=None,  # nunca loga por step
    )

    # Model
    model = ResNet(num_classes=4)

    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
