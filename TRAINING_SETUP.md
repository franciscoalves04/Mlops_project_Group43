# Enhanced Training System - Summary

## ✅ What's Been Added

### 1. **Hydra Configuration Management**
- Hierarchical config system in `conf/` directory
- Easy experiment tracking and hyperparameter management
- Command-line overrides without code changes

### 2. **WandB Integration**
- Automatic logging of metrics, hyperparameters, and system stats
- Model artifact tracking
- Training curve visualization
- Experiment comparison dashboard

### 3. **PyTorch Profiler**
- Performance profiling for optimization
- GPU/CPU utilization tracking
- Bottleneck identification

### 4. **Enhanced Callbacks**
- `MetricsCallback`: Track all training metrics
- `ModelArtifactCallback`: Save models to WandB
- `TrainingCurveCallback`: Auto-generate and save training plots
- `LearningRateMonitor`: Track LR changes

### 5. **Configuration Files**

```
conf/
├── config.yaml              # Main configuration
├── model/resnet.yaml        # Model architecture params
├── training/
│   ├── default.yaml        # Standard 30 epoch training
│   └── fast.yaml          # Quick 3 epoch testing
├── data/default.yaml        # Data configuration
└── experiment/
    ├── high_lr.yaml        # High LR experiment
    └── large_batch.yaml    # Large batch experiment
```

## 🚀 Usage Examples

### Basic Training
```bash
# Default config (30 epochs, batch=64, WandB enabled)
python src/eye_deseases_classification/train.py

# Quick test (3 epochs, no WandB)
python src/eye_deseases_classification/train.py training=fast logging.use_wandb=false
```

### Override Parameters
```bash
# Change batch size and learning rate
python src/eye_deseases_classification/train.py \
    training.batch_size=32 \
    model.learning_rate=0.0005

# Disable early stopping, train 50 epochs
python src/eye_deseases_classification/train.py \
    training.epochs=50 \
    training.early_stopping.enabled=false
```

### Run Experiments
```bash
# High learning rate experiment
python src/eye_deseases_classification/train.py +experiment=high_lr

# Large batch experiment
python src/eye_deseases_classification/train.py +experiment=large_batch
```

### Enable Profiling
```bash
python src/eye_deseases_classification/train.py profiling.enabled=true
```

## 📊 What Gets Logged

### Metrics
- Training loss and accuracy (per epoch)
- Validation loss and accuracy (per epoch)
- Learning rate schedule
- Best model checkpoint score

### Visualizations
- Training curves (loss + accuracy) saved to `reports/figures/`
- Logged to WandB dashboard

### Artifacts
- Best model checkpoint
- Final model state dict
- All configs used for the run

## 🔧 Setup

1. **Install dependencies:**
```bash
uv sync
```

2. **For GPU training (optional):**
```bash
uv pip install --force-reinstall torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. **Setup WandB (first time):**
```bash
wandb login
```

4. **Run training:**
```bash
.venv/bin/python src/eye_deseases_classification/train.py
```

## 📁 Output Structure

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/           # Hydra creates timestamped folders
        ├── .hydra/         # Saved configs
        ├── train.log       # Logs
        └── ...

models/                      # Best checkpoints
├── best_model-epoch=XX-val_acc=0.XXX.ckpt

logs/                        # CSV logs and profiler
├── eye_diseases_training/
└── profiler/

reports/figures/             # Training curves
└── training_curves.png
```

## 🎯 Benefits

1. **Reproducibility**: Every run's config is saved automatically
2. **Experiment Tracking**: Compare runs easily in WandB
3. **Flexibility**: Change hyperparameters without editing code
4. **Profiling**: Identify performance bottlenecks
5. **Visualization**: Auto-generated training curves
6. **Team Collaboration**: Share configs and results via WandB

## 📖 More Info

See detailed instructions in: `src/eye_deseases_classification/conf/README.md`
