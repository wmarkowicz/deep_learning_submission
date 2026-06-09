import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from typing import List
import random


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep="\t")
    df["sequence"] = df["sequence"].str[15:-15] # Removing flanking sequences.
    df = df[["sequence", "rna_dna_ratio", "is_active"]]
    return df

def split_data(df: pd.DataFrame, seed: int = 42, val_size: float = 0.1, test_size: float = 0.1) \
        -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=seed,
        stratify=df["is_active"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=temp_df["is_active"],
    )
    return train_df, val_df, test_df

def split_data_cross_validation(
    df: pd.DataFrame, seed: int, n_splits: int, test_size: float
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["is_active"],
    )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in splitter.split(train_val_df, train_val_df["is_active"]):
        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()
        folds.append((train_df, val_df, test_df.copy()))

    return folds

def normalize_rna_dna_ratio(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    mean = df["rna_dna_ratio"].mean()
    std = df["rna_dna_ratio"].std()
    df["rna_dna_ratio_norm"] = (df["rna_dna_ratio"] - mean) / std
    return df, mean, std

def denormalize_rna_dna_ratio(norm_values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (norm_values * std) + mean

class DNADataset(Dataset):
    def __init__(self, sequences: List[str], ratios: np.ndarray, actives: np.ndarray, seed = 42, augment: bool = False):
        acgt_map = {
            "A": [1, 0, 0, 0],
            "C": [0, 1, 0, 0],
            "G": [0, 0, 1, 0],
            "T": [0, 0, 0, 1],
        }
        self.sequences = np.array([[acgt_map[base] for base in seq] for seq in sequences], dtype=np.float32)
        self.ratios = np.asarray(ratios, dtype=np.float32)
        self.actives = np.asarray(actives, dtype=np.float32)
        self.rg = np.random.default_rng(seed)
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.augment and self.rg.random() < 0.5:
            complement_map = np.array([[0, 0, 0, 1],  # A -> T
                                       [0, 0, 1, 0],  # C -> G
                                       [0, 1, 0, 0],  # G -> C
                                       [1, 0, 0, 0]]) # T -> A
            seq = seq[::-1] @ complement_map
        return (
            torch.tensor(seq.T, dtype=torch.float32),
            torch.tensor([self.ratios[idx]], dtype=torch.float32),
            torch.tensor([self.actives[idx]], dtype=torch.float32),
        )

def make_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                ratio_mean: float, ratio_std: float, 
                 batch_size: int = 128, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = DNADataset(
        sequences=train_df["sequence"].tolist(),
        ratios=train_df["rna_dna_ratio_norm"].to_numpy(dtype=np.float32),
        actives=train_df["is_active"].to_numpy(dtype=np.float32),
        seed=seed,
        augment=True,
    )
    val_df['rna_dna_ratio_norm'] = (val_df['rna_dna_ratio'] - ratio_mean) / ratio_std
    val_ds = DNADataset(
        sequences=val_df["sequence"].tolist(),
        ratios=val_df["rna_dna_ratio_norm"].to_numpy(dtype=np.float32),
        actives=val_df["is_active"].to_numpy(dtype=np.float32),
        seed=seed,
        augment=False,
    )
    test_df['rna_dna_ratio_norm'] = (test_df['rna_dna_ratio'] - ratio_mean) / ratio_std
    test_ds = DNADataset(
        sequences=test_df["sequence"].tolist(),
        ratios=test_df["rna_dna_ratio_norm"].to_numpy(dtype=np.float32),
        actives=test_df["is_active"].to_numpy(dtype=np.float32),
        seed=seed,
        augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator= torch.Generator().manual_seed(seed) if seed is not None else None,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def make_cross_val_loaders(
    df: pd.DataFrame, n_splits: int, seed: int,
    normalize: bool, test_size: float = 0.1, batch_size: int = 128
    ) -> list[tuple[DataLoader, DataLoader, DataLoader, float | None, float | None]]:
    
    fold_loaders = []

    for train_df, val_df, test_df in split_data_cross_validation(
        df=df,
        seed=seed,
        n_splits=n_splits,
        test_size=test_size,
    ):
        mean, std = None, None
        if normalize:
            train_df, mean, std = normalize_rna_dna_ratio(train_df)

        train_loader, val_loader, test_loader = make_loaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            ratio_mean=mean,
            ratio_std=std,
            batch_size=batch_size,
            seed=seed,
        )
        fold_loaders.append((train_loader, val_loader, test_loader, mean, std))

    return fold_loaders

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reverse_complement_batch(batch):
    return batch[:, [3, 2, 1, 0], :].flip(-1)

def checkpoint_path_for_seed(checkpoint_dir: str, model_name: str, seed: int):
    return checkpoint_dir / f"{model_name}_{seed}.pth"

def metadata_path_for_seed(checkpoint_dir: str, model_name: str, seed: int):
    return checkpoint_dir / f"{model_name}_{seed}_meta.json"
