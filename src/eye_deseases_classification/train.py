import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from eye_deseases_classification.data import MyDataset
from eye_deseases_classification.model import ResNet


@hydra.main(
    config_name="config",
    config_path="../../configs",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Train ResNet model with Hydra configuration.

    Args:
        cfg: Hydra configuration object containing hyperparameters.
    """
    # Load datasets
    train_dataset = MyDataset(cfg.train_dataset_path)
    val_dataset = MyDataset(cfg.val_dataset_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k
    )

    early_stop_cb = EarlyStopping(
        monitor=cfg.early_stopping.monitor,
        patience=cfg.early_stopping.patience,
        mode=cfg.early_stopping.mode
    )

    # Logger CSV - garante que só grava uma linha por epoch
    logger = CSVLogger(save_dir=cfg.logger.save_dir, name=cfg.logger.name)

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.trainer.accelerator,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # Model
    model = ResNet(num_classes=cfg.model.num_classes)
    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
