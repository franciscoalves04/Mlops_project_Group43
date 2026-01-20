"""Model evaluation script with comprehensive metrics and visualizations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from eye_diseases_classification.data import MyDataset
from eye_diseases_classification.model import ResNet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""

    def __init__(
        self,
        checkpoint_path: str,
        test_data_path: str = "data/processed/test",
        batch_size: int = 32,
        device: str = "auto",
    ):
        """Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint (.ckpt file)
            test_data_path: Path to test dataset
            batch_size: Batch size for evaluation
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.test_data_path = test_data_path
        self.batch_size = batch_size

        # Load model
        log.info(f"Loading model from {self.checkpoint_path}")
        self.model = ResNet.load_from_checkpoint(self.checkpoint_path)
        self.model.eval()

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = self.model.to(self.device)
        log.info(f"Using device: {self.device}")

        # Load test data
        log.info(f"Loading test data from {self.test_data_path}")
        self.test_dataset = MyDataset(self.test_data_path)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # Class names
        self.class_names = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

        # Storage for predictions
        self.predictions: List[int] = []
        self.targets: List[int] = []
        self.probabilities: List[np.ndarray] = []

    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run predictions on test set.

        Returns:
            predictions: Predicted class indices
            targets: Ground truth class indices
            probabilities: Class probabilities for each sample
        """
        log.info("Running predictions on test set...")
        self.predictions = []
        self.targets = []
        self.probabilities = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                # Store results
                self.predictions.extend(preds.cpu().numpy())
                self.targets.extend(labels.cpu().numpy())
                self.probabilities.extend(probs.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    log.info(f"Processed {(batch_idx + 1) * self.batch_size}/{len(self.test_dataset)} samples")

        self.predictions = np.array(self.predictions)
        self.targets = np.array(self.targets)
        self.probabilities = np.array(self.probabilities)

        log.info(f"✅ Predictions complete: {len(self.predictions)} samples")
        return self.predictions, self.targets, self.probabilities

    def compute_metrics(self) -> Dict[str, float]:
        """Compute classification metrics.

        Returns:
            Dictionary of metrics
        """
        log.info("Computing metrics...")

        metrics = {
            "accuracy": accuracy_score(self.targets, self.predictions),
            "precision_macro": precision_score(self.targets, self.predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(self.targets, self.predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(self.targets, self.predictions, average="macro", zero_division=0),
            "precision_weighted": precision_score(self.targets, self.predictions, average="weighted", zero_division=0),
            "recall_weighted": recall_score(self.targets, self.predictions, average="weighted", zero_division=0),
            "f1_weighted": f1_score(self.targets, self.predictions, average="weighted", zero_division=0),
        }

        # Per-class metrics
        precision_per_class = precision_score(self.targets, self.predictions, average=None, zero_division=0)
        recall_per_class = recall_score(self.targets, self.predictions, average=None, zero_division=0)
        f1_per_class = f1_score(self.targets, self.predictions, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            metrics[f"precision_{class_name}"] = precision_per_class[i]
            metrics[f"recall_{class_name}"] = recall_per_class[i]
            metrics[f"f1_{class_name}"] = f1_per_class[i]

        return metrics

    def plot_confusion_matrix(self, save_path: str = "reports/figures/confusion_matrix.png") -> None:
        """Plot and save confusion matrix.

        Args:
            save_path: Path to save the confusion matrix plot
        """
        log.info("Generating confusion matrix...")

        cm = confusion_matrix(self.targets, self.predictions)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        log.info(f"Confusion matrix saved to {save_path}")

    def plot_per_class_metrics(self, save_path: str = "reports/figures/per_class_metrics.png") -> None:
        """Plot per-class precision, recall, and F1 scores.

        Args:
            save_path: Path to save the plot
        """
        log.info("Generating per-class metrics plot...")

        # Compute per-class metrics
        precision = precision_score(self.targets, self.predictions, average=None, zero_division=0)
        recall = recall_score(self.targets, self.predictions, average=None, zero_division=0)
        f1 = f1_score(self.targets, self.predictions, average=None, zero_division=0)

        # Create plot
        x = np.arange(len(self.class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label="Precision", color="skyblue")
        ax.bar(x, recall, width, label="Recall", color="lightcoral")
        ax.bar(x + width, f1, width, label="F1 Score", color="lightgreen")

        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis="y", alpha=0.3)

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        log.info(f"Per-class metrics plot saved to {save_path}")

    def save_report(self, save_path: str = "reports/evaluation_report.txt") -> None:
        """Save detailed evaluation report to text file.

        Args:
            save_path: Path to save the report
        """
        log.info("Generating evaluation report...")

        metrics = self.compute_metrics()

        # Create report content
        report_lines = [
            "=" * 80,
            "MODEL EVALUATION REPORT",
            "=" * 80,
            f"Model: {self.checkpoint_path}",
            f"Test Dataset: {self.test_data_path}",
            f"Test Samples: {len(self.test_dataset)}",
            f"Device: {self.device}",
            "",
            "=" * 80,
            "OVERALL METRICS",
            "=" * 80,
            f"Accuracy: {metrics['accuracy']:.4f}",
            "",
            "Macro Averages:",
            f"  Precision: {metrics['precision_macro']:.4f}",
            f"  Recall:    {metrics['recall_macro']:.4f}",
            f"  F1 Score:  {metrics['f1_macro']:.4f}",
            "",
            "Weighted Averages:",
            f"  Precision: {metrics['precision_weighted']:.4f}",
            f"  Recall:    {metrics['recall_weighted']:.4f}",
            f"  F1 Score:  {metrics['f1_weighted']:.4f}",
            "",
            "=" * 80,
            "PER-CLASS METRICS",
            "=" * 80,
        ]

        # Per-class metrics table
        for class_name in self.class_names:
            report_lines.append(f"\n{class_name.upper()}:")
            report_lines.append(f"  Precision: {metrics[f'precision_{class_name}']:.4f}")
            report_lines.append(f"  Recall:    {metrics[f'recall_{class_name}']:.4f}")
            report_lines.append(f"  F1 Score:  {metrics[f'f1_{class_name}']:.4f}")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("CLASSIFICATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(classification_report(self.targets, self.predictions, target_names=self.class_names))
        report_lines.append("=" * 80)

        # Save report
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write("\n".join(report_lines))

        log.info(f"Evaluation report saved to {save_path}")

        # Also print to console
        print("\n".join(report_lines))

    def evaluate_full(self) -> Dict[str, float]:
        """Run full evaluation pipeline.

        Returns:
            Dictionary of metrics
        """
        log.info("=" * 80)
        log.info("Starting full evaluation pipeline")
        log.info("=" * 80)

        # Run predictions
        self.predict()

        # Compute metrics
        metrics = self.compute_metrics()

        # Generate visualizations
        self.plot_confusion_matrix()
        self.plot_per_class_metrics()

        # Save report
        self.save_report()

        log.info("=" * 80)
        log.info("✅ Evaluation complete!")
        log.info("=" * 80)

        return metrics


def evaluate(checkpoint_path: Optional[str] = None, test_data_path: str = "data/processed/test") -> Dict[str, float]:
    """Main evaluation function.

    Args:
        checkpoint_path: Path to checkpoint. If None, finds best checkpoint in models/
        test_data_path: Path to test dataset

    Returns:
        Dictionary of evaluation metrics
    """
    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_dir = Path("models")
        ckpts = sorted(checkpoint_dir.glob("best_model*.ckpt"))

        if len(ckpts) == 0:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}/")

        checkpoint_path = str(ckpts[-1])  # Use most recent
        log.info(f"Using checkpoint: {checkpoint_path}")

    # Create evaluator and run
    evaluator = ModelEvaluator(checkpoint_path=checkpoint_path, test_data_path=test_data_path)
    metrics = evaluator.evaluate_full()

    return metrics


if __name__ == "__main__":
    evaluate()
