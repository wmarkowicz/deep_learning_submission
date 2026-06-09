"""Model-agnostic, consensus-driven beam search for DNA sequence optimization.

This generalizes ``domi_wika_beam_search.py``:

* Any group member's regression model is wrapped behind a single interface
  (``Scorer.predict(sequences) -> np.ndarray`` in real rna_dna_ratio units).
  Adding/removing a model is just editing the ``--models`` list.
* The optimization objective is a *consensus* across all selected models
  (default: maximize the minimum per-model improvement), so a design only wins
  if every model agrees it is better. This is the most robust target against
  the hidden external benchmark model.
* Stopping uses *patience* instead of breaking on the first non-improving step,
  so the search can climb out of small plateaus and actually spend its budget.

The Domi (5-fold DeepSTARRRDSEB ensemble) and Wika (CNN + LightGBM hybrid)
scorers are copied verbatim from the validated two-model script. The Ola
scorer (k-mer LightGBM regressor) is new and faithfully reproduces the
``predicted_rna_dna_ratio`` that her evaluation_script.py emits.

Example
-------
    python task_2_solution/sequence_optimization/multi_model_beam_search.py \
        --models domi wika ola --objective min --beam-width 4 --patience 3
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _looks_like_project_root(path):
    return (
        (path / "task_materials").exists()
        and (path / "task_1_solution_domi").exists()
        and ((path / "task_2_solution").exists() or (path / "task_B").exists())
    )


def _resolve_project_root():
    """Find the original project root even when these scripts are copied out."""
    script_path = Path(__file__).resolve()
    candidates = []

    env_root = os.environ.get("DLS_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    for parent in script_path.parents:
        candidates.extend([parent, parent / "dls_project", parent / "deep_learning_submission"])

    for candidate in candidates:
        if _looks_like_project_root(candidate):
            return candidate

    # Preserve the original layout assumption so failures point to the old paths.
    return script_path.parents[2]


ROOT = _resolve_project_root()
TASK2_DIR = ROOT / "task_2_solution"
if not TASK2_DIR.exists():
    TASK2_DIR = ROOT / "task_B"
DOMI_DIR = ROOT / "task_1_solution_domi"
DOMI_MODEL_DIR = DOMI_DIR / "models" / "DeepSTARRRDSEB"
WIKA_DIR = ROOT / "task_1_solution_wika"
OLA_DIR = ROOT / "task_1_solution_ola"
OLA_MODEL_DIR = OLA_DIR / "final_model"

sys.path.insert(0, str(DOMI_DIR))
from src.models import DeepSTARRRDSEB  # noqa: E402

BASES = "ACGT"
BASE_INDEX = {base: index for index, base in enumerate(BASES)}

# Mutation budgets. Subtask A budgets are fixed by the task. Subtask B has
# "no budget"; we cap it generously for tractability + biological plausibility.
DEFAULT_BUDGETS = {
    "seq1_broken": 40,
    "seq_2_broken": 16,
    "seq_3_broken": 20,
    "seq_4_no_active": 40,
}

# Adapter regions (first/last 15 nt) are shared constant flanks across all
# sequences and are invisible to the Domi model (which trims [15:-15]). We
# never mutate inside them.
CORE_START = 15
CORE_END = 215  # exclusive

def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_fasta(path):
    records = {}
    text = Path(path).read_text().strip()
    for entry in text.split(">"):
        if not entry.strip():
            continue
        lines = entry.splitlines()
        record_id = lines[0].strip()
        sequence = "".join(line.strip() for line in lines[1:] if line.strip()).upper()
        records[record_id] = sequence
    return records


def load_records(paths):
    records = {}
    for path in paths:
        records.update(read_fasta(path))
    return records


def reverse_complement_batch(batch):
    return batch[:, [3, 2, 1, 0], :].flip(-1)


def mutation_count(original_sequence, sequence):
    return sum(
        original_base != new_base
        for original_base, new_base in zip(original_sequence, sequence)
    )


def _gradient_position_gains(path):
    """Per-position best single-substitution gain from Domi's gradient CSV."""
    rows = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            current = row["current_base"]
            current_score = float(row[f"gradient_estimated_score_if_{current}"])
            best_gain = max(
                float(row[f"gradient_estimated_score_if_{base}"]) - current_score
                for base in BASES
                if base != current
            )
            rows.append((best_gain, int(row["full_position"]) - 1))
    return rows


def _ism_position_gains(path):
    """Per-position best consensus (min_delta) gain from an ISM CSV."""
    best = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            position = int(row["full_position"]) - 1
            gain = float(row["min_delta"])
            if position not in best or gain > best[position]:
                best[position] = gain
    return [(gain, position) for position, gain in best.items()]


def load_priority_positions(prior, gradient_dir, ism_dir, record_id, top_positions):
    """Return the top candidate positions for beam search.

    prior='gradient' -> cheap single-model approximation (Domi).
    prior='ism'      -> exact, consensus across all models (gradient -> ism -> beam).
    """
    if top_positions is None or top_positions <= 0:
        return None
    if prior == "ism":
        path = ism_dir / f"{record_id}_ism.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"ISM prior requested but {path} is missing. "
                f"Run ism_attribution.py for {record_id} first."
            )
        rows = _ism_position_gains(path)
    else:
        path = gradient_dir / f"{record_id}_attribution.csv"
        if not path.exists():
            return None
        rows = _gradient_position_gains(path)
    rows.sort(reverse=True)
    return [position for _, position in rows[:top_positions]]


def single_mutants(sequence, used_positions, priority_positions):
    if priority_positions is not None:
        positions = priority_positions
    else:
        positions = range(CORE_START, CORE_END)
    for position in positions:
        if position in used_positions:
            continue
        current_base = sequence[position]
        for new_base in BASES:
            if new_base == current_base:
                continue
            mutant = sequence[:position] + new_base + sequence[position + 1:]
            yield {
                "position": position,
                "from": current_base,
                "to": new_base,
                "sequence": mutant,
            }


def write_tsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_budget_overrides(spec):
    """Parse optional record-specific budgets, e.g. seq_4_no_active=80."""
    overrides = {}
    for item in spec or []:
        if "=" not in item:
            raise ValueError(f"Invalid budget override '{item}', expected id=value")
        record_id, value = item.split("=", 1)
        overrides[record_id] = int(value)
    return overrides

class DomiSequenceDataset(Dataset):
    def __init__(self, sequences):
        core = [sequence.upper()[15:-15] for sequence in sequences]
        encoded = np.zeros((len(core), 4, len(core[0])), dtype=np.float32)
        for row, sequence in enumerate(core):
            for position, base in enumerate(sequence):
                encoded[row, BASE_INDEX[base], position] = 1.0
        self.encoded = encoded

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        return torch.tensor(self.encoded[index], dtype=torch.float32)


class DomiScorer:
    """5-fold DeepSTARRRDSEB ensemble, reverse-complement TTA, real units."""

    name = "domi"

    def __init__(self, model_dir, device, batch_size):
        self.model_dir = Path(model_dir)
        self.device = device
        self.batch_size = batch_size
        self.fold_stats = json.loads((self.model_dir / "fold_stats.json").read_text())
        self.models = []
        for fold in range(1, len(self.fold_stats) + 1):
            model = DeepSTARRRDSEB().to(self.device)
            checkpoint = self.model_dir / f"model_fold{fold}_seed42.pt"
            model.load_state_dict(
                torch.load(checkpoint, map_location=self.device, weights_only=True)
            )
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def predict(self, sequences):
        dataset = DomiSequenceDataset(sequences)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        fold_predictions = []
        for model, stat in zip(self.models, self.fold_stats):
            predictions = []
            for batch in loader:
                batch = batch.to(self.device)
                pred = model(batch)
                pred_rc = model(reverse_complement_batch(batch))
                pred = 0.5 * (pred + pred_rc)
                pred = pred.squeeze(-1).cpu().numpy()
                pred = pred * float(stat["ratio_std"]) + float(stat["ratio_mean"])
                predictions.append(pred)
            fold_predictions.append(np.concatenate(predictions))
        return np.mean(fold_predictions, axis=0)


class WikaScorer:
    """CNN + LightGBM hybrid regressor with RC-TTA. Returns regression only."""

    name = "wika"

    def __init__(self, model_dir, device, batch_size):
        self.eval = load_module(
            Path(model_dir) / "evaluation_script.py", "wika_evaluation_script"
        )
        self.model_dir = Path(model_dir)
        self.device = device
        self.batch_size = batch_size
        self.config = json.loads((self.model_dir / "final_config.json").read_text())

        self.seq_len = int(self.config["seq_len"])
        self.adapter_5 = int(self.config.get("adapter_5", 0))
        self.adapter_3 = int(self.config.get("adapter_3", 0))
        self.use_scaled_reg = bool(self.config.get("use_scaled_reg", True))
        self.reg_mean = float(self.config.get("reg_mean", 0.0))
        self.reg_std = float(self.config.get("reg_std", 1.0))
        self.reg_alpha = float(self.config["regression"]["cnn_lgb_alpha"])

        checkpoint = torch.load(
            self.model_dir / self.config["model_paths"]["cnn"],
            map_location=self.device,
            weights_only=False,
        )
        self.model = self.eval.DeepSTARRInspired(
            seq_len=int(checkpoint["model_config"]["seq_len"]),
            dropout=float(checkpoint["model_config"]["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.lgb_reg = joblib.load(
            self.model_dir / self.config["model_paths"]["lgb_regressor"]
        )

    def predict(self, sequences):
        sequences = [sequence.upper() for sequence in sequences]
        sequences_rc = [self.eval.reverse_complement(s) for s in sequences]

        _, reg_orig = self.eval.predict_cnn_outputs(
            self.model, sequences, self.seq_len, self.device, self.batch_size,
            self.adapter_5, self.adapter_3, self.use_scaled_reg,
            self.reg_mean, self.reg_std,
        )
        _, reg_rc = self.eval.predict_cnn_outputs(
            self.model, sequences_rc, self.seq_len, self.device, self.batch_size,
            self.adapter_5, self.adapter_3, self.use_scaled_reg,
            self.reg_mean, self.reg_std,
        )
        reg_cnn = 0.5 * reg_orig + 0.5 * reg_rc

        x_feats = self.eval.extract_features(
            self.model, sequences, self.seq_len, self.device,
            self.batch_size, self.adapter_5, self.adapter_3,
        )
        x_feats_rc = self.eval.extract_features(
            self.model, sequences_rc, self.seq_len, self.device,
            self.batch_size, self.adapter_5, self.adapter_3,
        )
        if bool(self.config.get("use_handcrafted_dna_features", True)):
            x_dna = self.eval.dna_feature_matrix(
                sequences, self.seq_len, self.adapter_5, self.adapter_3
            )
            x_dna_rc = self.eval.dna_feature_matrix(
                sequences_rc, self.seq_len, self.adapter_5, self.adapter_3
            )
            x_feats = np.hstack([x_feats, x_dna])
            x_feats_rc = np.hstack([x_feats_rc, x_dna_rc])

        reg_lgb = 0.5 * self.lgb_reg.predict(x_feats)
        reg_lgb += 0.5 * self.lgb_reg.predict(x_feats_rc)

        final_reg = self.reg_alpha * reg_cnn + (1.0 - self.reg_alpha) * reg_lgb
        return np.asarray(final_reg, dtype=float)


class OlaScorer:
    """k-mer LightGBM regressor. Reproduces Ola's predicted_rna_dna_ratio.
    """

    name = "ola"

    def __init__(self, model_dir, device, batch_size):
        self.model_dir = Path(model_dir)
        self.vectorizer = joblib.load(self.model_dir / "kmer_vectorizer.pkl")
        self.lgb_reg = joblib.load(self.model_dir / "lgbm_regressor.pkl")

    def predict(self, sequences):
        sequences = [str(s).upper() for s in sequences]
        features = self.vectorizer.transform(sequences).astype(np.float32)
        return np.asarray(self.lgb_reg.predict(features), dtype=float)


SCORER_REGISTRY = {
    "domi": (DomiScorer, DOMI_MODEL_DIR),
    "wika": (WikaScorer, WIKA_DIR),
    "ola": (OlaScorer, OLA_MODEL_DIR),
}

class ConsensusScorer:
    """Scores a batch against every selected model and reports per-model deltas
    versus the original sequence, plus consensus summaries."""

    def __init__(self, scorers, original_sequence):
        self.scorers = scorers
        self.names = [scorer.name for scorer in scorers]
        self.original = original_sequence
        self.original_scores = {
            scorer.name: float(scorer.predict([original_sequence])[0])
            for scorer in scorers
        }

    def predict(self, sequences):
        per_model = {}
        deltas = {}
        for scorer in self.scorers:
            scores = np.asarray(scorer.predict(sequences), dtype=float)
            per_model[scorer.name] = scores
            deltas[scorer.name] = scores - self.original_scores[scorer.name]
        delta_stack = np.vstack([deltas[name] for name in self.names])
        return {
            "reg": per_model,
            "delta": deltas,
            "min_delta": delta_stack.min(axis=0),
            "mean_delta": delta_stack.mean(axis=0),
        }


def objective_value(min_delta, mean_delta, deltas, index, mode, floor, penalty, nmut):
    if mode == "min":
        value = float(min_delta)
    elif mode == "mean":
        value = float(mean_delta)
    elif mode == "mean_floor":
        value = float(mean_delta)
        for name in deltas:
            d = float(deltas[name][index])
            if d < floor:
                value -= 1000.0 * (floor - d)
    else:
        raise ValueError(f"unknown objective mode: {mode}")
    return value - penalty * nmut

def history_entry(record_id, step, scorer_names, scores, index, objective,
                  mutation, position, frm, to, mutations_used, sequence):
    entry = {
        "id": record_id,
        "step": step,
        "objective": objective,
        "min_delta": float(scores["min_delta"][index]),
        "mean_delta": float(scores["mean_delta"][index]),
        "mutation": mutation,
        "position": position,
        "from": frm,
        "to": to,
        "mutations_used": mutations_used,
    }
    for name in scorer_names:
        entry[f"reg_{name}"] = float(scores["reg"][name][index])
        entry[f"delta_{name}"] = float(scores["delta"][name][index])
    entry["sequence"] = sequence
    return entry


def beam_optimize(consensus, record_id, sequence, budget, priority_positions,
                  beam_width, patience, objective_mode, floor, penalty):
    names = consensus.names
    init_scores = consensus.predict([sequence])
    init_obj = objective_value(
        init_scores["min_delta"][0], init_scores["mean_delta"][0],
        init_scores["delta"], 0, objective_mode, floor, penalty, 0,
    )
    init_history = [
        history_entry(record_id, 0, names, init_scores, 0, init_obj,
                      "original", "", "", "", 0, sequence)
    ]
    beam = [{
        "sequence": sequence,
        "objective": init_obj,
        "used_positions": set(),
        "history": init_history,
    }]
    best = beam[0]
    no_improve = 0

    for step in range(1, budget + 1):
        candidate_meta = []
        seen = set()
        for parent_index, parent in enumerate(beam):
            for candidate in single_mutants(
                parent["sequence"], parent["used_positions"], priority_positions
            ):
                if candidate["sequence"] in seen:
                    continue
                seen.add(candidate["sequence"])
                candidate_meta.append((parent_index, candidate))
        if not candidate_meta:
            break

        candidate_sequences = [c["sequence"] for _, c in candidate_meta]
        scores = consensus.predict(candidate_sequences)

        children = []
        for index, (parent_index, candidate) in enumerate(candidate_meta):
            parent = beam[parent_index]
            used = set(parent["used_positions"])
            used.add(candidate["position"])
            obj = objective_value(
                scores["min_delta"][index], scores["mean_delta"][index],
                scores["delta"], index, objective_mode, floor, penalty, len(used),
            )
            full_position = candidate["position"] + 1
            mutation = f"{full_position}:{candidate['from']}->{candidate['to']}"
            entry = history_entry(
                record_id, step, names, scores, index, obj, mutation,
                full_position, candidate["from"], candidate["to"],
                len(used), candidate["sequence"],
            )
            children.append({
                "sequence": candidate["sequence"],
                "objective": obj,
                "used_positions": used,
                "history": parent["history"] + [entry],
            })

        children.sort(key=lambda row: row["objective"], reverse=True)
        beam = children[:beam_width]

        if beam[0]["objective"] > best["objective"] + 1e-9:
            best = beam[0]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best["sequence"], best["history"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["domi", "wika", "ola"],
        choices=list(SCORER_REGISTRY), help="Which member models form the consensus.",
    )
    parser.add_argument(
        "--objective", default="min", choices=["min", "mean", "mean_floor"],
        help="min = maximize the worst per-model gain (most robust).",
    )
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--top-positions", type=int, default=80,
                        help="Candidate positions from attribution prior; 0 = scan whole core.")
    parser.add_argument("--prior", default="gradient", choices=["gradient", "ism"],
                        help="Source of candidate positions for beam search.")
    parser.add_argument("--mutation-penalty", type=float, default=0.0)
    parser.add_argument("--floor", type=float, default=0.0,
                        help="Per-model floor used only by --objective mean_floor.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--budget-overrides", nargs="*", default=None,
        help="Override mutation budgets, e.g. seq_4_no_active=80 seq1_broken=40.",
    )
    parser.add_argument(
        "--fasta", nargs="+", type=Path,
        default=[ROOT / "task_materials" / "subtaskA.fa",
                 ROOT / "task_materials" / "subtaskB.fa"],
    )
    parser.add_argument(
        "--attribution-dir", type=Path,
        default=TASK2_DIR / "gradient_attribution" / "outputs",
        help="Gradient-attribution CSV directory (used when --prior gradient).",
    )
    parser.add_argument(
        "--ism-dir", type=Path,
        default=TASK2_DIR / "ism_scanning" / "outputs",
        help="ISM CSV directory (used when --prior ism).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(args.fasta)
    budgets = dict(DEFAULT_BUDGETS)
    budgets.update(parse_budget_overrides(args.budget_overrides))

    print(f"device: {device}")
    print(f"models: {args.models} | objective: {args.objective} | "
          f"beam: {args.beam_width} | patience: {args.patience} | prior: {args.prior}")

    scorers = []
    for name in args.models:
        scorer_cls, model_dir = SCORER_REGISTRY[name]
        scorers.append(scorer_cls(model_dir, device, args.batch_size))

    tag = "_".join(args.models) + f"_{args.objective}_{args.prior}"
    output_dir = args.output_dir or (
        TASK2_DIR / "sequence_optimization" / "outputs" / tag
    )

    results = []
    all_history = []
    for record_id, budget in budgets.items():
        if record_id not in records:
            continue
        sequence = records[record_id]
        priority_positions = load_priority_positions(
            args.prior, args.attribution_dir, args.ism_dir, record_id, args.top_positions
        )
        consensus = ConsensusScorer(scorers, sequence)
        new_sequence, history = beam_optimize(
            consensus, record_id, sequence, budget, priority_positions,
            args.beam_width, args.patience, args.objective,
            args.floor, args.mutation_penalty,
        )
        final = history[-1]
        all_history.extend(history)

        consensus_pred = float(np.mean(
            [final[f"reg_{name}"] for name in consensus.names]
        ))
        result = {
            "id": record_id,
            "mutations_used": len(history) - 1,
            "mutation_budget": budget,
            "objective": final["objective"],
            "min_delta": final["min_delta"],
            "mean_delta": final["mean_delta"],
            "predicted_rna_dna_ratio": consensus_pred,
        }
        for name in consensus.names:
            result[f"original_{name}"] = consensus.original_scores[name]
            result[f"final_{name}"] = final[f"reg_{name}"]
            result[f"delta_{name}"] = final[f"delta_{name}"]
        result["new_sequence"] = new_sequence
        results.append(result)

        deltas_str = ", ".join(
            f"{name} {consensus.original_scores[name]:+.3f}->{final[f'reg_{name}']:+.3f}"
            for name in consensus.names
        )
        print(f"{record_id}: {deltas_str} | min_delta={final['min_delta']:.3f} "
              f"| {len(history) - 1}/{budget} muts", flush=True)

    # Detailed result table.
    detail_fields = ["id", "mutations_used", "mutation_budget", "objective",
                     "min_delta", "mean_delta", "predicted_rna_dna_ratio"]
    for name in args.models:
        detail_fields += [f"original_{name}", f"final_{name}", f"delta_{name}"]
    detail_fields.append("new_sequence")
    write_tsv(output_dir / "optimized_sequences_detailed.tsv", results, detail_fields)

    history_fields = ["id", "step", "objective", "min_delta", "mean_delta"]
    for name in args.models:
        history_fields += [f"reg_{name}", f"delta_{name}"]
    history_fields += ["mutation", "position", "from", "to",
                       "mutations_used", "sequence"]
    write_tsv(output_dir / "optimization_history.tsv", all_history, history_fields)

    deliverable = [
        {"id": r["id"], "new_sequence": r["new_sequence"],
         "predicted_rna_dna_ratio": r["predicted_rna_dna_ratio"]}
        for r in results
    ]
    write_tsv(output_dir / "submission.tsv", deliverable,
              ["id", "new_sequence", "predicted_rna_dna_ratio"])

    print(f"\nsaved to: {output_dir}")


if __name__ == "__main__":
    main()
