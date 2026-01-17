# Training Configuration Guide

## Quick Start

Install dependencies:
```bash
uv sync
```

### Basic Training

Run with default configuration:
```bash
uv run python src/eye_deseases_classification/train.py
```

Or directly if venv is activated:
```bash
python src/eye_deseases_classification/train.py
```

### Configuration Overrides

Override specific parameters from command line:

```bash
# Change experiment name
python src/eye_deseases_classification/train.py experiment.name="my-experiment"

# Change batch size and epochs
python src/eye_deseases_classification/train.py training.batch_size=32 training.epochs=50

# Change learning rate
python src/eye_deseases_classification/train.py model.learning_rate=0.0005

# Disable early stopping
python src/eye_deseases_classification/train.py training.early_stopping.enabled=false

# Disable WandB logging (use only CSV)
python src/eye_deseases_classification/train.py logging.use_wandb=false

# Enable profiling
python src/eye_deseases_classification/train.py profiling.enabled=true
```

### Using Different Training Configs

Fast training (3 epochs for testing):
```bash
python src/eye_deseases_classification/train.py training=fast
```

### Running Experiments

Use predefined experiment configs:

```bash
# High learning rate experiment
python src/eye_deseases_classification/train.py +experiment=high_lr

# Large batch size experiment
python src/eye_deseases_classification/train.py +experiment=large_batch
```

### Multiple Runs

Run multiple experiments with different seeds:
```bash
for seed in 42 123 456; do
    python src/eye_deseases_classification/train.py experiment.seed=$seed experiment.name="run-seed-$seed"
done
```

### Hydra Outputs

Hydra creates output directories with timestamps:
```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/          # Config files
        ├── train.log        # Training logs
        └── ...
```

To disable this and use fixed paths:
```bash
python src/eye_deseases_classification/train.py hydra.output_subdir=null hydra.run.dir=.
```

## Configuration Structure

```
conf/
├── config.yaml              # Main config
├── model/
│   └── resnet.yaml         # Model architecture
├── training/
│   ├── default.yaml        # Standard training
│   └── fast.yaml          # Quick testing
├── data/
│   └── default.yaml        # Data config
└── experiment/
    ├── high_lr.yaml        # Experiment: high LR
    └── large_batch.yaml    # Experiment: large batch
```

## WandB Integration

First time setup:
```bash
wandb login
```

WandB will track:
- Training/validation loss and accuracy
- Learning rate schedule
- Training curves visualization
- Model artifacts
- System metrics (GPU, CPU, memory)

View runs at: https://wandb.ai

## Profiling

Enable PyTorch profiler:
```bash
python src/eye_deseases_classification/train.py profiling.enabled=true
```

View trace file:
```bash
tensorboard --logdir=logs/profiler
```

## Tips

1. **Test configs quickly**: Use `training=fast` for 3 epochs
2. **Sweep hyperparameters**: Use WandB sweeps or Hydra multirun
3. **Save experiments**: Each run's config is saved automatically by Hydra
4. **Compare runs**: Use WandB dashboard to compare metrics
5. **Resume training**: Load from checkpoint and continue

## Example Workflow

```bash
# 1. Quick test with fast training
python src/eye_deseases_classification/train.py training=fast logging.use_wandb=false

# 2. Full training with WandB
python src/eye_deseases_classification/train.py experiment.name="baseline-run1"

# 3. Experiment with hyperparameters
python src/eye_deseases_classification/train.py +experiment=high_lr

# 4. Check profiling
python src/eye_deseases_classification/train.py profiling.enabled=true training=fast
```
