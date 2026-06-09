
import sys
import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

class ReverseComplementCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),

            nn.Conv1d(96, 160, kernel_size=7, padding=3),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            nn.Conv1d(160, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),

            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.shared = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        self.classification_head = nn.Linear(128, 1)
        self.regression_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)

        x_max = self.global_max_pool(x).squeeze(-1)
        x_avg = self.global_avg_pool(x).squeeze(-1)

        x = torch.cat([x_max, x_avg], dim=1)
        x = self.shared(x)

        cls_logits = self.classification_head(x).squeeze(1)
        reg_output = self.regression_head(x).squeeze(1)

        return cls_logits, reg_output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def safe_std(x):
    if x == 0:
        return 1.0
    return x


def preprocess_sequence(seq):
    return str(seq).upper()


def reverse_complement(seq):
    complement = str.maketrans("ACGT", "TGCA")
    return seq.translate(complement)[::-1]


def one_hot_encode(seq):
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
    }

    arr = np.array(
        [mapping.get(base, [0, 0, 0, 0]) for base in seq],
        dtype=np.float32
    )

    return arr.T


def predict_cnn_probs(model, sequences, device, batch_size=512):
    model.eval()

    all_probs = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start:start + batch_size]

            # Forward strand
            x_forward = np.stack([one_hot_encode(seq) for seq in batch_seqs])
            x_forward = torch.tensor(x_forward, dtype=torch.float32).to(device)

            logits_forward, _ = model(x_forward)
            probs_forward = torch.sigmoid(logits_forward).cpu().numpy()

            # Reverse complement strand
            batch_rc = [reverse_complement(seq) for seq in batch_seqs]
            x_rc = np.stack([one_hot_encode(seq) for seq in batch_rc])
            x_rc = torch.tensor(x_rc, dtype=torch.float32).to(device)

            logits_rc, _ = model(x_rc)
            probs_rc = torch.sigmoid(logits_rc).cpu().numpy()

            # Test-time augmentation average
            probs = (probs_forward + probs_rc) / 2

            all_probs.extend(probs)

    return np.array(all_probs)

def main():
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python evaluation_script.py <path_to_model> <path_to_test_data>"
        )

    model_dir = sys.argv[1]
    test_path = sys.argv[2]

    with open(os.path.join(model_dir, "final_config.json"), "r") as f:
        config = json.load(f)

    vectorizer = joblib.load(os.path.join(model_dir, "kmer_vectorizer.pkl"))
    ridge = joblib.load(os.path.join(model_dir, "ridge_regressor.pkl"))
    lgbm_clf = joblib.load(os.path.join(model_dir, "lgbm_classifier.pkl"))
    lgbm_reg = joblib.load(os.path.join(model_dir, "lgbm_regressor.pkl"))

    # Logistic Regression may exist, but final config can set its weight to 0.
    logreg_path = os.path.join(model_dir, "logreg_classifier.pkl")
    logreg = None
    if os.path.exists(logreg_path):
        logreg = joblib.load(logreg_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_checkpoint = torch.load(
        os.path.join(model_dir, "rc_cnn_model.pth"),
        map_location=device
    )

    cnn_model = ReverseComplementCNN().to(device)
    cnn_model.load_state_dict(cnn_checkpoint["model_state_dict"])
    cnn_model.eval()

    test_df = pd.read_csv(test_path, sep="\t")

    if "id" not in test_df.columns and "seq_id" in test_df.columns:
        test_df = test_df.rename(columns={"seq_id": "id"})

    if "id" not in test_df.columns or "sequence" not in test_df.columns:
        raise ValueError("Test TSV must contain columns: id, sequence")

    sequences = test_df["sequence"].apply(preprocess_sequence).tolist()

    X_test = vectorizer.transform(sequences)
    X_test_lgb = X_test.astype(np.float32)

    cnn_probs = predict_cnn_probs(
        cnn_model,
        sequences,
        device=device,
        batch_size=512
    )

    lgbm_clf_probs = lgbm_clf.predict_proba(X_test_lgb)[:, 1]

    ridge_preds = ridge.predict(X_test)
    lgbm_reg_preds = lgbm_reg.predict(X_test_lgb)

    # optional logreg
    if logreg is not None:
        logreg_probs = logreg.predict_proba(X_test)[:, 1]
    else:
        logreg_probs = np.zeros_like(cnn_probs)

    # Convert regression predictions into probability-like scores
    ridge_std = safe_std(float(config["ridge_pred_std"]))
    ridge_score = (
        ridge_preds - float(config["ridge_pred_mean"])
    ) / ridge_std
    ridge_prob_like = sigmoid(ridge_score)

    lgbm_reg_std = safe_std(float(config["lgbm_reg_pred_std"]))
    lgbm_reg_score = (
        lgbm_reg_preds - float(config["lgbm_reg_pred_mean"])
    ) / lgbm_reg_std
    lgbm_reg_prob_like = sigmoid(lgbm_reg_score)

    w_cnn = float(config.get("w_cnn", 0.5))
    w_lgbm_clf = float(config.get("w_lgbm_clf", 0.2))
    w_logreg = float(config.get("w_logreg", 0.0))
    w_lgbm_reg = float(config.get("w_lgbm_reg", 0.0))
    w_ridge = float(config.get("w_ridge", 0.3))

    ensemble_probs = (
        w_cnn * cnn_probs
        + w_lgbm_clf * lgbm_clf_probs
        + w_logreg * logreg_probs
        + w_lgbm_reg * lgbm_reg_prob_like
        + w_ridge * ridge_prob_like
    )

    threshold = float(config.get("classification_threshold", 0.41))

    predicted_is_active = (ensemble_probs >= threshold).astype(int)

    predicted_rna_dna_ratio = lgbm_reg_preds

    print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")

    for seq_id, pred_cls, pred_reg in zip(
        test_df["id"].values,
        predicted_is_active,
        predicted_rna_dna_ratio
    ):
        print(f"{seq_id}\t{int(pred_cls)}\t{float(pred_reg)}")


if __name__ == "__main__":
    main()
