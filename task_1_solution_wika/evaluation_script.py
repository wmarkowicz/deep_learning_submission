"""
Evaluation script for the Regulatory Activity of DNA project.

Usage:
    python evaluation_script.py <path_to_folder_with_config_and_models> <path_to_test_data>

Example:
    python evaluation_script.py . test_data.tsv

The first argument must be a path to the folder containing:
    - final_config.json
    - best_model_deep.pth
    - lgb_classifier.pkl
    - lgb_regressor.pkl

The test data should be a TSV file with columns:
    - id
    - sequence

The script loads the saved configuration and models, generates predictions,
and prints them to stdout in the required TSV format:
    id    predicted_is_active    predicted_rna_dna_ratio
"""
import os
import sys
import json
import joblib
import warnings
from itertools import product

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Model architecture
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.1):
        super().__init__()

        pad = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class DeepSTARRInspired(nn.Module):
    def __init__(self, seq_len=230, dropout=0.2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        self.tower = nn.Sequential(
            ConvBlock(256, 256, kernel_size=7, dilation=1, dropout=dropout),
            nn.MaxPool1d(2),

            ConvBlock(256, 512, kernel_size=5, dilation=1, dropout=dropout),
            ConvBlock(512, 512, kernel_size=5, dilation=2, dropout=dropout),
            nn.MaxPool1d(2),

            ConvBlock(512, 512, kernel_size=3, dilation=1, dropout=dropout),
            ConvBlock(512, 512, kernel_size=3, dilation=4, dropout=dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.shared_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.tower(x)
        x = self.gap(x).squeeze(-1)
        x = self.shared_fc(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        cls_logit = self.cls_head(x).squeeze(-1)
        reg_out = self.reg_head(x).squeeze(-1)
        return cls_logit, reg_out


# =========================
# Preprocessing
# =========================

DNA_MAPPING = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}

RC_MAP = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq):
    return seq.upper().translate(RC_MAP)[::-1]


def trim_adapters(seq, adapter_5=0, adapter_3=0):
    seq = str(seq)
    if adapter_3 == 0:
        return seq[adapter_5:]
    return seq[adapter_5: len(seq) - adapter_3]


def one_hot_encode(seq, length):
    seq = str(seq).upper()
    ohe = np.zeros((4, length), dtype=np.float32)

    for i, base in enumerate(seq[:length]):
        idx = DNA_MAPPING.get(base)
        if idx is not None:
            ohe[idx, i] = 1.0

    return ohe


class PredDataset(Dataset):
    def __init__(self, sequences, seq_len, adapter_5=0, adapter_3=0):
        self.sequences = sequences
        self.seq_len = seq_len
        self.adapter_5 = adapter_5
        self.adapter_3 = adapter_3

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = trim_adapters(self.sequences[idx], self.adapter_5, self.adapter_3)
        x = torch.from_numpy(one_hot_encode(seq, self.seq_len)).float()
        return x


# =========================
# Handcrafted DNA features
# =========================

KMER_LISTS = {
    2: ["".join(p) for p in product("ACGT", repeat=2)],
    3: ["".join(p) for p in product("ACGT", repeat=3)],
    4: ["".join(p) for p in product("ACGT", repeat=4)],
}

KMER_INDEX = {
    k: {mer: i for i, mer in enumerate(kmers)}
    for k, kmers in KMER_LISTS.items()
}


def dna_feature_matrix(sequences, seq_len, adapter_5=0, adapter_3=0):
    n_basic = 5
    n_kmers = sum(len(v) for v in KMER_LISTS.values())
    X = np.zeros((len(sequences), n_basic + n_kmers), dtype=np.float32)

    for row, seq in enumerate(sequences):
        seq = trim_adapters(seq, adapter_5, adapter_3).upper()[:seq_len]
        L = max(len(seq), 1)

        count_a = seq.count("A")
        count_c = seq.count("C")
        count_g = seq.count("G")
        count_t = seq.count("T")

        X[row, 0] = count_a / L
        X[row, 1] = count_c / L
        X[row, 2] = count_g / L
        X[row, 3] = count_t / L
        X[row, 4] = (count_g + count_c) / L

        offset = n_basic

        for k, kmers in KMER_LISTS.items():
            denom = max(L - k + 1, 1)
            idx_map = KMER_INDEX[k]

            for i in range(L - k + 1):
                mer = seq[i:i + k]
                idx = idx_map.get(mer)
                if idx is not None:
                    X[row, offset + idx] += 1.0 / denom

            offset += len(kmers)

    return X


# =========================
# Prediction helpers
# =========================

@torch.no_grad()
def predict_cnn_outputs(model, sequences, seq_len, device, batch_size, adapter_5, adapter_3, use_scaled_reg, reg_mean, reg_std):
    ds = PredDataset(sequences, seq_len, adapter_5, adapter_3)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    all_reg = []

    model.eval()

    for x in dl:
        x = x.to(device)
        logits, reg_out = model(x)

        probs = torch.sigmoid(logits).cpu().numpy()
        reg_pred = reg_out.cpu().numpy()

        if use_scaled_reg:
            reg_pred = reg_pred * reg_std + reg_mean

        all_probs.append(probs)
        all_reg.append(reg_pred)

    return np.concatenate(all_probs), np.concatenate(all_reg)


@torch.no_grad()
def extract_features(model, sequences, seq_len, device, batch_size, adapter_5, adapter_3):
    ds = PredDataset(sequences, seq_len, adapter_5, adapter_3)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats = []

    model.eval()

    for x in dl:
        x = x.to(device)
        f = model.forward_features(x)
        feats.append(f.cpu().numpy())

    return np.vstack(feats)


def resolve_path(model_dir, path_from_config):
    if os.path.isabs(path_from_config):
        return path_from_config
    return os.path.join(model_dir, path_from_config)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python evaluation_script.py <path_to_folder_with_config_and_models> <path_to_test_data>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_arg = sys.argv[1]
    test_path = sys.argv[2]

    # The first argument can be either:
    # 1. a folder containing final_config.json and model files
    # 2. final_config.json itself
    # 3. one of the model files located in the same folder as final_config.json
    if os.path.isdir(model_arg):
        model_dir = model_arg
        config_path = os.path.join(model_dir, "final_config.json")
    else:
        model_dir = os.path.dirname(model_arg) or "."
        if os.path.basename(model_arg) == "final_config.json":
            config_path = model_arg
        else:
            config_path = os.path.join(model_dir, "final_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    seq_len = int(config["seq_len"])
    adapter_5 = int(config.get("adapter_5", 0))
    adapter_3 = int(config.get("adapter_3", 0))
    use_scaled_reg = bool(config.get("use_scaled_reg", True))
    reg_mean = float(config.get("reg_mean", 0.0))
    reg_std = float(config.get("reg_std", 1.0))

    cls_threshold = float(config["classification"]["threshold"])
    cls_alpha = float(config["classification"]["main_cnn_lgb_alpha"])
    reg_alpha = float(config["regression"]["cnn_lgb_alpha"])

    cnn_path = resolve_path(model_dir, config["model_paths"]["cnn"])
    lgb_cls_path = resolve_path(model_dir, config["model_paths"]["lgb_classifier"])
    lgb_reg_path = resolve_path(model_dir, config["model_paths"]["lgb_regressor"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)

    model = DeepSTARRInspired(
        seq_len=int(checkpoint["model_config"]["seq_len"]),
        dropout=float(checkpoint["model_config"]["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    lgb_cls = joblib.load(lgb_cls_path)
    lgb_reg = joblib.load(lgb_reg_path)

    test_df = pd.read_csv(test_path, sep="\t")

    if "id" in test_df.columns:
        id_col = "id"
    elif "seq_id" in test_df.columns:
        id_col = "seq_id"
    else:
        raise ValueError("Test file must contain an 'id' column.")

    if "sequence" not in test_df.columns:
        raise ValueError("Test file must contain a 'sequence' column.")

    ids = test_df[id_col].tolist()
    sequences = test_df["sequence"].astype(str).str.upper().tolist()
    sequences_rc = [reverse_complement(s) for s in sequences]

    batch_size = 256

    # CNN TTA predictions
    probs_orig, reg_orig = predict_cnn_outputs(
        model, sequences, seq_len, device, batch_size,
        adapter_5, adapter_3, use_scaled_reg, reg_mean, reg_std
    )

    probs_rc, reg_rc = predict_cnn_outputs(
        model, sequences_rc, seq_len, device, batch_size,
        adapter_5, adapter_3, use_scaled_reg, reg_mean, reg_std
    )

    probs_cnn = 0.5 * probs_orig + 0.5 * probs_rc
    reg_cnn = 0.5 * reg_orig + 0.5 * reg_rc

    # CNN embeddings + handcrafted DNA features for LightGBM
    X_feats = extract_features(
        model, sequences, seq_len, device, batch_size, adapter_5, adapter_3
    )

    X_feats_rc = extract_features(
        model, sequences_rc, seq_len, device, batch_size, adapter_5, adapter_3
    )

    if bool(config.get("use_handcrafted_dna_features", True)):
        X_dna = dna_feature_matrix(sequences, seq_len, adapter_5, adapter_3)
        X_dna_rc = dna_feature_matrix(sequences_rc, seq_len, adapter_5, adapter_3)

        X_feats = np.hstack([X_feats, X_dna])
        X_feats_rc = np.hstack([X_feats_rc, X_dna_rc])

    probs_lgb_orig = lgb_cls.predict_proba(X_feats)[:, 1]
    probs_lgb_rc = lgb_cls.predict_proba(X_feats_rc)[:, 1]
    probs_lgb = 0.5 * probs_lgb_orig + 0.5 * probs_lgb_rc

    reg_lgb_orig = lgb_reg.predict(X_feats)
    reg_lgb_rc = lgb_reg.predict(X_feats_rc)
    reg_lgb = 0.5 * reg_lgb_orig + 0.5 * reg_lgb_rc

    # Final ensemble
    final_probs = cls_alpha * probs_cnn + (1.0 - cls_alpha) * probs_lgb
    final_reg = reg_alpha * reg_cnn + (1.0 - reg_alpha) * reg_lgb

    final_cls = (final_probs >= cls_threshold).astype(int)

    # Required stdout format
    print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")
    for sample_id, pred_cls, pred_reg in zip(ids, final_cls, final_reg):
        print(f"{sample_id}\t{int(pred_cls)}\t{float(pred_reg):.8f}")


if __name__ == "__main__":
    main()
