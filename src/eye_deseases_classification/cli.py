import typer
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from eye_deseases_classification.train import train as train_fn
from eye_deseases_classification.evaluate import evaluate as evaluate_fn

app = typer.Typer(help="Eye diseases classification CLI")


@app.command()
def train(
    config_name: str = typer.Option(
        "config",
        "--config",
        "-c",
        help="Hydra config file name (without .yaml)",
    ),
    epochs: int = typer.Option(
        None,
        "--epochs",
        "-e",
        help="Override number of training epochs",
    ),
    batch_size: int = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Override batch size",
    ),
    learning_rate: float = typer.Option(
        None,
        "--learning-rate",
        "-lr",
        help="Override learning rate",
    ),
) -> None:
    """Train the ResNet model with Hydra configuration."""
    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).parent.parent.parent / "configs")

    try:
        with initialize_config_dir(version_base="1.2", config_dir=config_dir):
            overrides = []
            if epochs is not None:
                overrides.append(f"experiment.training.epochs={epochs}")
            if batch_size is not None:
                overrides.append(f"experiment.training.batch_size={batch_size}")
            if learning_rate is not None:
                overrides.append(f"experiment.training.learning_rate={learning_rate}")

            cfg: DictConfig = compose(config_name=config_name, overrides=overrides)
            OmegaConf.set_struct(cfg, False)
            typer.echo("Configuration loaded successfully")
            typer.echo(OmegaConf.to_yaml(cfg))
            train_fn(cfg)
    except Exception as e:
        typer.echo(f"Error loading config: {e}", err=True)
        typer.echo("Make sure configs/config.yaml exists and includes defaults", err=True)
        raise typer.Exit(1)


@app.command()
def evaluate(
    model_checkpoint: Path = typer.Option(
        Path("models/resnet_model.pt"),
        "--model",
        "-m",
        help="Path to model checkpoint",
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="Batch size for evaluation",
    ),
) -> None:
    """Evaluate the model on test dataset."""
    try:
        evaluate_fn(model_checkpoint=model_checkpoint, batch_size=batch_size)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()