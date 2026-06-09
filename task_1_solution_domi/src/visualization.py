import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import json
from pathlib import Path
import torch
from torch import nn

def load_model(model_class, ckpt_path: Path, device: torch.device) \
    -> tuple[nn.Module, dict]:
    model_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = model_class()
    model.load_state_dict(model_state)
    model.to(device)
    return model

def print_metrics(metrics: dict, set_name: str) -> None:
    print(f"\n===== {set_name} =====")
    print(f"Loss:     {metrics['loss']:.4f}")
    print(f"MSE:      {metrics['mse']:.4f}")
    print(f"Pearson:  {metrics['pearson']:.4f}")
    print(f"Spearman: {metrics['spearman']:.4f}")
    print(f"Accuracy: {metrics['acc']:.4f}")

def plot_confusion_matrix(y_true_a, y_pred_a):
    cm = confusion_matrix(y_true_a, y_pred_a)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Inactive", "Active"],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

def plot_regression(y_true_r, y_pred_r):
    lims = [min(y_true_r.min(), y_pred_r.min()), max(y_true_r.max(), y_pred_r.max())]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_r, y_pred_r, alpha=0.2)
    plt.plot(lims, lims, "r--", linewidth=2)
    plt.xlabel("True rna_dna_ratio")
    plt.ylabel("Predicted rna_dna_ratio")
    plt.title("Regression predictions")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.show()

def plot_violin(y_true_a, y_pred_r):
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x=y_true_a,
        y=y_pred_r,
        hue=y_true_a,
        legend=False,
        palette="Set2",
    )
    plt.axhline(0, color="red", linestyle="--", alpha=0.5)
    plt.title("Does the model separate classes in terms of predicted ratio?")
    plt.xlabel("True class (is_active)")
    plt.ylabel("Predicted rna_dna_ratio")
    plt.show()

def plot_threshold_curve(thresholds, accuracies, best_threshold):
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies)
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"best = {best_threshold:.3f}")
    plt.xlabel("Threshold on predicted ratio")
    plt.ylabel("Accuracy")
    plt.title("Threshold tuning")
    plt.legend()
    plt.show()
