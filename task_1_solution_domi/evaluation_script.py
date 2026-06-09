import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import denormalize_rna_dna_ratio
from src.models import DeepSTARRRDSEB
from src.train_utils import predict_ratio
from src.visualization import load_model

TRIM_FLANKS = 15
BATCH_SIZE = 128

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        acgt_map = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
        }
        self.sequences = np.array(
            [[[acgt_map[base] for base in sequence]] for sequence in sequences],
            dtype=np.float32,
        )[:, 0]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx].T, dtype=torch.float32)

def load_fold_stats(model_dir):
    stats_path = model_dir / "fold_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing file: {stats_path}")
    stats = json.loads(stats_path.read_text())
    return [(row["ratio_mean"], row["ratio_std"]) for row in stats]

def load_test_loader(test_data_path):
    test_df = pd.read_csv(test_data_path, sep="\t")
    test_df["sequence"] = test_df["sequence"].str[TRIM_FLANKS:-TRIM_FLANKS]
    dataset = SequenceDataset(test_df["sequence"].tolist())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_df, loader

def predict_one_model(model_path, ratio_mean, ratio_std, loader, device):
    model = load_model(DeepSTARRRDSEB, model_path, device)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            pred_ratio = predict_ratio(model, batch_x, use_rc_tta=True)
            preds.extend(pred_ratio.cpu().numpy().flatten())
    preds = np.array(preds)
    return denormalize_rna_dna_ratio(preds, ratio_mean, ratio_std)

if len(sys.argv) != 3:
    raise SystemExit("Usage: python evaluation_script.py <path_to_model> <path_to_test_data>")

model_dir = Path(sys.argv[1])
test_data_path = Path(sys.argv[2])

if not model_dir.exists():
    raise FileNotFoundError(f"Model path does not exist: {model_dir}")
if not model_dir.is_dir():
    raise ValueError(f"Model path must be a directory with fold checkpoints: {model_dir}")

device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")
test_df, test_loader = load_test_loader(test_data_path)
fold_stats = load_fold_stats(model_dir)

all_preds = []
for fold_idx, (mean, std) in enumerate(fold_stats, start=1):
    model_path = model_dir / f"model_fold{fold_idx}_seed42.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    preds = predict_one_model(model_path, mean, std, test_loader, device)
    all_preds.append(preds)

pred_rna_dna_ratio = np.mean(all_preds, axis=0)
pred_is_active = (pred_rna_dna_ratio > 0).astype(int)

output_df = pd.DataFrame(
    {
        "id": test_df["id"],
        "predicted_is_active": pred_is_active,
        "predicted_rna_dna_ratio": pred_rna_dna_ratio,
    }
)
output_df.to_csv(sys.stdout, sep="\t", index=False, header=False)