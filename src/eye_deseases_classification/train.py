from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from eye_deseases_classification.logger import logger

from eye_deseases_classification.data import MyDataset
from eye_deseases_classification.model import ResNet


def main():
    logger.info("Loading datasets...")
    train_dataset = MyDataset("data/processed/train")
    val_dataset = MyDataset("data/processed/val")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    logger.info("Datasets loaded successfully")

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

    logger.info("Initializing trainer...")
    logger.info("Model training started")
    logger_csv = CSVLogger(save_dir="logs", name="resnet_training")
    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger_csv,
        log_every_n_steps=None
    )

    model = ResNet(num_classes=4)
    trainer.fit(model, train_loader, val_loader)
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
