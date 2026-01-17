import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import DataLoader

from eye_deseases_classification.data import MyDataset
from eye_deseases_classification.model import ResNet

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

log = logging.getLogger(__name__)


class MetricsCallback(Callback):
    """Track metrics for plotting."""

    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics["train_loss"].append(trainer.callback_metrics.get("train_loss", 0).item())
        self.metrics["train_acc"].append(trainer.callback_metrics.get("train_acc", 0).item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics["val_loss"].append(trainer.callback_metrics.get("val_loss", 0).item())
        self.metrics["val_acc"].append(trainer.callback_metrics.get("val_acc", 0).item())


class ModelArtifactCallback(Callback):
    """Log model as wandb artifact."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    def on_fit_end(self, trainer, pl_module):
        if not WANDB_AVAILABLE or not wandb.run:
            log.info("WandB not available, skipping artifact upload")
            return

        # Save final model state dict
        final_model_path = os.path.join(self.model_dir, "final_model.pth")
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(pl_module.state_dict(), final_model_path)

        # Create artifact with both final model and best checkpoint
        artifact = wandb.Artifact(
            name="eye-diseases-model",
            type="model",
            description="Trained ResNet model on eye diseases dataset"
        )
        
        # Add final model state dict
        artifact.add_file(final_model_path, name="final_model.pth")
        
        # Add best checkpoint if it exists
        if hasattr(trainer.checkpoint_callback, 'best_model_path') and trainer.checkpoint_callback.best_model_path:
            best_ckpt_path = trainer.checkpoint_callback.best_model_path
            if os.path.exists(best_ckpt_path):
                artifact.add_file(best_ckpt_path, name="best_checkpoint.ckpt")
                log.info(f"Added best checkpoint to artifact: {best_ckpt_path}")
        
        wandb.log_artifact(artifact)
        log.info(f"Model artifacts saved to WandB: {artifact.name}")


class TrainingCurveCallback(Callback):
    """Plot and log training curves."""

    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir

    def on_fit_end(self, trainer, pl_module):
        # Get metrics from MetricsCallback
        metrics_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, MetricsCallback):
                metrics_callback = callback
                break

        if metrics_callback is None or not metrics_callback.metrics["train_loss"]:
            return

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(metrics_callback.metrics["train_loss"], label="Train Loss")
        ax1.plot(metrics_callback.metrics["val_loss"], label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(metrics_callback.metrics["train_acc"], label="Train Accuracy")
        ax2.plot(metrics_callback.metrics["val_acc"], label="Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        fig_path = os.path.join(self.reports_dir, "training_curves.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')

        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({"training_curves": wandb.Image(fig)})

        plt.close(fig)
        log.info(f"Training curves saved to {fig_path}")


@hydra.main(version_base=None, config_path="../eye_deseases_classification/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a model on eye diseases dataset."""
    log.info("=" * 80)
    log.info("Starting training for eye diseases classification")
    log.info("=" * 80)
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    torch.manual_seed(cfg.experiment.seed)

    # Initialize model with config parameters
    model = ResNet(
        num_classes=cfg.model.num_classes,
        lr=cfg.model.learning_rate
    )

    # Load data
    train_dataset = MyDataset(
        cfg.paths.data_dir + "/train",
        augment=cfg.data.augmentation.train
    )
    val_dataset = MyDataset(
        cfg.paths.data_dir + "/val",
        augment=cfg.data.augmentation.val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.hardware.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        pin_memory=True
    )

    log.info(f"Training samples: {len(train_dataset)}")
    log.info(f"Validation samples: {len(val_dataset)}")

    # Callbacks
    callbacks = []

    # Metrics tracking
    metrics_callback = MetricsCallback()
    callbacks.append(metrics_callback)

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.training.checkpoint.monitor,
        dirpath=cfg.paths.model_dir,
        filename=cfg.training.checkpoint.filename,
        save_top_k=cfg.training.checkpoint.save_top_k,
        mode=cfg.training.checkpoint.mode,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.training.early_stopping.enabled:
        early_stopping_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            verbose=True
        )
        callbacks.append(early_stopping_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Training curves
    curve_callback = TrainingCurveCallback(reports_dir=cfg.paths.reports_dir)
    callbacks.append(curve_callback)

    # Model artifact callback (for wandb)
    if cfg.logging.use_wandb and WANDB_AVAILABLE:
        artifact_callback = ModelArtifactCallback(model_dir=cfg.paths.model_dir)
        callbacks.append(artifact_callback)

    # Loggers
    loggers = []

    # CSV logger
    if cfg.logging.use_csv:
        csv_logger = CSVLogger(
            save_dir=cfg.logging.csv_save_dir,
            name="eye_diseases_training"
        )
        loggers.append(csv_logger)

    # WandB logger
    if cfg.logging.use_wandb:
        if not WANDB_AVAILABLE:
            log.warning("WandB requested but not installed. Run: uv add wandb")
        else:
            wandb_logger = WandbLogger(
                project=cfg.logging.wandb_project,
                name=cfg.experiment.name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            loggers.append(wandb_logger)

    # Profiler
    profiler = None
    if cfg.profiling.enabled:
        profiler = PyTorchProfiler(
            dirpath=cfg.profiling.dirpath,
            filename=cfg.profiling.filename
        )
        log.info("PyTorch profiler enabled")

    # Trainer
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        callbacks=callbacks,
        logger=loggers if loggers else None,
        profiler=profiler,
        deterministic=True,
        log_every_n_steps=None,
    )

    # Train
    log.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Log final metrics
    log.info("=" * 80)
    log.info("Training completed!")
    log.info(f"Best model saved to: {checkpoint_callback.best_model_path}")
    log.info(f"Best {cfg.training.checkpoint.monitor}: {checkpoint_callback.best_model_score:.4f}")
    log.info("=" * 80)

    # Finish wandb run
    if cfg.logging.use_wandb and WANDB_AVAILABLE and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
