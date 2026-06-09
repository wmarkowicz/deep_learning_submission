import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy import stats
import json
from torch.nn import functional as F

from src.data import *
from src.models import *
from src.visualization import *

def build_model(seed: int, model_class: type[nn.Module], device: torch.device,
    learning_rate: float, weight_decay: float) ->  \
    tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    seed_everything(seed)
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    return model, optimizer, scheduler

def train(seed: int,
          model: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
          device: torch.device,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int,
          patience: int,
          ckpt_path: Path,
          meta_path: Path,
          mean: float | None = None,
          std: float | None = None,
    ) -> None:
    best_score = -np.inf
    best_metrics = None
    epochs_without_improvement = 0

    print(f"=== TRAINING SEED {seed} ===")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for input_batch, rna_dna_ratio, _active in train_loader:
            input_batch = input_batch.to(device)
            rna_dna_ratio = rna_dna_ratio.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_ratio = model(input_batch)
            loss = F.smooth_l1_loss(pred_ratio, rna_dna_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            normalized_prediction_used=mean is not None and std is not None,
            mean=mean,
            std=std,
            use_rc_tta=False,
        )
        val_score = val_metrics["pearson"] + val_metrics["acc"] - 0.10 * val_metrics["mse"]
        scheduler.step(val_score)

        print(
            f"Seed {seed} | Epoch {epoch+1:02d} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val MSE: {val_metrics['mse']:.4f} | "
            f"Val Pearson: {val_metrics['pearson']:.4f} | "
            f"Val Spearman: {val_metrics['spearman']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} | "
            f"Val Score: {val_score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_score > best_score:
            best_score = val_score
            best_metrics = {
                "seed": seed,
                "epoch": epoch + 1,
                "score": float(best_score),
                "val_loss": float(val_metrics["loss"]),
                "mse": float(val_metrics["mse"]),
                "pearson": float(val_metrics["pearson"]),
                "spearman": float(val_metrics["spearman"]),
                "acc": float(val_metrics["acc"]),
            }
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
            meta_path.write_text(json.dumps(best_metrics, indent=2))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Seed {seed}: early stopping at epoch {epoch+1}")
            break

    if best_metrics is not None:
        print(f"Seed {seed}: best epoch {best_metrics['epoch']} saved")


def predict_ratio(model, batch_x, use_rc_tta=False):
    pred = model(batch_x)
    if use_rc_tta:
        pred_rc = model(reverse_complement_batch(batch_x))
        pred = 0.5 * (pred + pred_rc)
    return pred

def evaluate_model(
        model, 
        loader, 
        device: torch.device,
        normalized_prediction_used: bool,
        mean: float | None = None,
        std: float | None = None,
        loss_fn: nn.Module = nn.SmoothL1Loss(),
        use_rc_tta=False,
    ) -> dict[str, float | np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true_rna_dna_ratio = []
    y_pred_rna_dna_ratio = []
    y_true_active = []
    y_pred_active = []

    with torch.no_grad():
        for input_batch, rna_dna_ratio, active in loader:
            input_batch = input_batch.to(device)
            rna_dna_ratio = rna_dna_ratio.to(device)

            pred_ratio = predict_ratio(model, input_batch, use_rc_tta=use_rc_tta)
            loss = loss_fn(pred_ratio, rna_dna_ratio)
            total_loss += loss.item()
            if normalized_prediction_used:
                if mean is None or std is None:
                    raise ValueError("mean and std are required when normalized_prediction_used=True")
                y_true_rna_dna_ratio.extend(
                    denormalize_rna_dna_ratio(rna_dna_ratio.cpu().numpy().flatten(), mean, std)
                )
                y_pred_rna_dna_ratio.extend(
                    denormalize_rna_dna_ratio(pred_ratio.cpu().numpy().flatten(), mean, std)
                )
            else:
                y_true_rna_dna_ratio.extend(rna_dna_ratio.cpu().numpy().flatten())
                y_pred_rna_dna_ratio.extend(pred_ratio.cpu().numpy().flatten())    
            y_true_active.extend(active.cpu().numpy().flatten())

    y_true_rna_dna_ratio = np.array(y_true_rna_dna_ratio)
    y_pred_rna_dna_ratio = np.array(y_pred_rna_dna_ratio)
    y_true_active = np.array(y_true_active)
    y_pred_active = (y_pred_rna_dna_ratio > 0).astype(int)

    return {
        "loss": total_loss / len(loader),
        "mse": mean_squared_error(y_true_rna_dna_ratio, y_pred_rna_dna_ratio),
        "pearson": stats.pearsonr(y_true_rna_dna_ratio, y_pred_rna_dna_ratio)[0],
        "spearman": stats.spearmanr(y_true_rna_dna_ratio, y_pred_rna_dna_ratio)[0],
        "acc": accuracy_score(y_true_active, y_pred_active),
        "y_true_rna_dna_ratio": y_true_rna_dna_ratio,
        "y_pred_rna_dna_ratio": y_pred_rna_dna_ratio,
        "y_true_active": y_true_active,
        "y_pred_active": y_pred_active,
    }
