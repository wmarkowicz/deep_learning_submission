"""Rank-weighted, consensus-driven beam search for DNA sequence optimization.

This is a small experimental wrapper around ``multi_model_beam_search.py``.
It reuses the same FASTA parsing, attribution priors and model scorers, but
adds a rank-weighted objective. The motivation is that the group models did
not perform equally in the individual benchmark, so the search can give more
influence to the stronger model while still keeping the remaining models as
regularizers.

Example
-------
    python task_2_solution/sequence_optimization/weighted_multi_model_beam_search.py \
        --models domi wika ola \
        --weights domi=0.5 wika=0.3 ola=0.2 \
        --objective weighted_mean_floor \
        --floor 3.15 \
        --prior ism \
        --beam-width 8 \
        --patience 10 \
        --top-positions 160 \
        --fasta task_materials/subtaskB.fa \
        --budget-overrides seq_4_no_active=80
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_model_beam_search as mb  # noqa: E402


def parse_model_weights(spec, model_names):
    """Parse ``name=value`` weights and normalize over selected models."""
    if not spec:
        return {name: 1.0 / len(model_names) for name in model_names}

    weights = {}
    for item in spec:
        if "=" not in item:
            raise ValueError(f"Invalid model weight '{item}', expected name=value")
        name, value = item.split("=", 1)
        weights[name] = float(value)

    missing = [name for name in model_names if name not in weights]
    if missing:
        raise ValueError(f"Missing weights for selected models: {missing}")

    total = sum(weights[name] for name in model_names)
    if total <= 0:
        raise ValueError("Model weights must sum to a positive value")
    return {name: weights[name] / total for name in model_names}


def parse_budget_overrides(spec):
    overrides = {}
    for item in spec or []:
        if "=" not in item:
            raise ValueError(f"Invalid budget override '{item}', expected id=value")
        record_id, value = item.split("=", 1)
        overrides[record_id] = int(value)
    return overrides


def objective_value(scores, index, names, model_weights, mode, floor, penalty, nmut):
    deltas = {name: float(scores["delta"][name][index]) for name in names}
    min_delta = min(deltas.values())
    mean_delta = float(np.mean(list(deltas.values())))
    weighted_delta = sum(model_weights[name] * deltas[name] for name in names)

    if mode == "min":
        value = min_delta
    elif mode == "mean":
        value = mean_delta
    elif mode == "weighted_mean":
        value = weighted_delta
    elif mode == "mean_floor":
        value = mean_delta
        for delta in deltas.values():
            if delta < floor:
                value -= 1000.0 * (floor - delta)
    elif mode == "weighted_mean_floor":
        value = weighted_delta
        for name, delta in deltas.items():
            if delta < floor:
                # The floor is a hard safety constraint, so all selected models
                # are protected regardless of their ranking weight.
                value -= 1000.0 * (floor - delta)
    else:
        raise ValueError(f"unknown objective mode: {mode}")

    return value - penalty * nmut, weighted_delta


def weighted_value(values, names, model_weights):
    return sum(model_weights[name] * float(values[name]) for name in names)


def history_entry(record_id, step, names, model_weights, scores, index, objective,
                  weighted_delta, weighted_prediction, mutation, position, frm,
                  to, mutations_used, sequence):
    entry = {
        "id": record_id,
        "step": step,
        "objective": objective,
        "min_delta": float(scores["min_delta"][index]),
        "mean_delta": float(scores["mean_delta"][index]),
        "weighted_delta": weighted_delta,
        "weighted_prediction": weighted_prediction,
        "mutation": mutation,
        "position": position,
        "from": frm,
        "to": to,
        "mutations_used": mutations_used,
    }
    for name in names:
        entry[f"weight_{name}"] = model_weights[name]
        entry[f"reg_{name}"] = float(scores["reg"][name][index])
        entry[f"delta_{name}"] = float(scores["delta"][name][index])
    entry["sequence"] = sequence
    return entry


def beam_optimize(consensus, record_id, sequence, budget, priority_positions,
                  beam_width, patience, objective_mode, floor, penalty,
                  model_weights):
    names = consensus.names
    init_scores = consensus.predict([sequence])
    init_obj, init_weighted_delta = objective_value(
        init_scores, 0, names, model_weights, objective_mode, floor, penalty, 0
    )
    init_history = [
        history_entry(record_id, 0, names, model_weights, init_scores, 0,
                      init_obj, init_weighted_delta,
                      weighted_value(
                          {name: init_scores["reg"][name][0] for name in names},
                          names,
                          model_weights,
                      ),
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
            for candidate in mb.single_mutants(
                parent["sequence"], parent["used_positions"], priority_positions
            ):
                if candidate["sequence"] in seen:
                    continue
                seen.add(candidate["sequence"])
                candidate_meta.append((parent_index, candidate))
        if not candidate_meta:
            break

        candidate_sequences = [candidate["sequence"] for _, candidate in candidate_meta]
        scores = consensus.predict(candidate_sequences)

        children = []
        for index, (parent_index, candidate) in enumerate(candidate_meta):
            parent = beam[parent_index]
            used = set(parent["used_positions"])
            used.add(candidate["position"])
            obj, weighted_delta = objective_value(
                scores, index, names, model_weights, objective_mode, floor,
                penalty, len(used)
            )
            weighted_prediction = weighted_value(
                {name: scores["reg"][name][index] for name in names},
                names,
                model_weights,
            )
            full_position = candidate["position"] + 1
            mutation = f"{full_position}:{candidate['from']}->{candidate['to']}"
            entry = history_entry(
                record_id, step, names, model_weights, scores, index, obj,
                weighted_delta, weighted_prediction, mutation, full_position,
                candidate["from"], candidate["to"], len(used),
                candidate["sequence"]
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
    parser.add_argument("--models", nargs="+", default=["domi", "wika", "ola"],
                        choices=list(mb.SCORER_REGISTRY))
    parser.add_argument("--model-weights", "--weights", nargs="*", default=None,
                        help="Model weights, e.g. domi=0.5 wika=0.3 ola=0.2")
    parser.add_argument("--prediction-report", default="weighted",
                        choices=["weighted", "mean"],
                        help="Which ensemble prediction to place in submission.tsv.")
    parser.add_argument("--objective", default="weighted_mean_floor",
                        choices=["min", "mean", "weighted_mean",
                                 "mean_floor", "weighted_mean_floor"])
    parser.add_argument("--floor", type=float, default=0.0)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--top-positions", type=int, default=80,
                        help="Candidate positions from attribution prior; 0 = whole core.")
    parser.add_argument("--prior", default="ism", choices=["gradient", "ism"])
    parser.add_argument("--mutation-penalty", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--records", nargs="+", default=None,
                        help="Optional subset of FASTA record ids.")
    parser.add_argument("--budget-overrides", nargs="*", default=None,
                        help="Override budgets, e.g. seq_4_no_active=80")
    parser.add_argument(
        "--fasta", nargs="+", type=Path,
        default=[mb.ROOT / "task_materials" / "subtaskA.fa",
                 mb.ROOT / "task_materials" / "subtaskB.fa"],
    )
    parser.add_argument(
        "--attribution-dir", type=Path,
        default=mb.TASK2_DIR / "gradient_attribution" / "outputs",
    )
    parser.add_argument(
        "--ism-dir", type=Path,
        default=mb.TASK2_DIR / "ism_scanning" / "outputs",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = mb.load_records(args.fasta)
    target_records = set(args.records) if args.records else None
    budgets = dict(mb.DEFAULT_BUDGETS)
    budgets.update(parse_budget_overrides(args.budget_overrides))

    print(f"device: {device}")
    print(f"models: {args.models} | objective: {args.objective} | "
          f"beam: {args.beam_width} | patience: {args.patience} | prior: {args.prior}")

    scorers = []
    for name in args.models:
        scorer_cls, model_dir = mb.SCORER_REGISTRY[name]
        scorers.append(scorer_cls(model_dir, device, args.batch_size))

    model_weights = parse_model_weights(args.model_weights, args.models)
    print(f"model weights: {model_weights}")

    tag = "weighted_" + "_".join(args.models) + f"_{args.objective}_{args.prior}"
    output_dir = args.output_dir or (
        mb.TASK2_DIR / "sequence_optimization" / "outputs" / tag
    )

    results = []
    all_history = []
    for record_id, budget in budgets.items():
        if record_id not in records:
            continue
        if target_records is not None and record_id not in target_records:
            continue

        sequence = records[record_id]
        priority_positions = mb.load_priority_positions(
            args.prior, args.attribution_dir, args.ism_dir, record_id,
            args.top_positions
        )
        consensus = mb.ConsensusScorer(scorers, sequence)
        new_sequence, history = beam_optimize(
            consensus, record_id, sequence, budget, priority_positions,
            args.beam_width, args.patience, args.objective, args.floor,
            args.mutation_penalty, model_weights,
        )
        final = history[-1]
        all_history.extend(history)

        unweighted_pred = float(np.mean(
            [final[f"reg_{name}"] for name in consensus.names]
        ))
        weighted_pred = sum(
            model_weights[name] * final[f"reg_{name}"] for name in consensus.names
        )
        consensus_pred = (
            weighted_pred if args.prediction_report == "weighted"
            else unweighted_pred
        )
        result = {
            "id": record_id,
            "mutations_used": len(history) - 1,
            "mutation_budget": budget,
            "objective": final["objective"],
            "min_delta": final["min_delta"],
            "mean_delta": final["mean_delta"],
            "weighted_delta": final["weighted_delta"],
            "unweighted_predicted_rna_dna_ratio": unweighted_pred,
            "weighted_predicted_rna_dna_ratio": weighted_pred,
            "predicted_rna_dna_ratio": consensus_pred,
        }
        for name in consensus.names:
            result[f"weight_{name}"] = model_weights[name]
            result[f"original_{name}"] = consensus.original_scores[name]
            result[f"final_{name}"] = final[f"reg_{name}"]
            result[f"delta_{name}"] = final[f"delta_{name}"]
        result["new_sequence"] = new_sequence
        results.append(result)

        deltas_str = ", ".join(
            f"{name} {consensus.original_scores[name]:+.3f}->{final[f'reg_{name}']:+.3f}"
            for name in consensus.names
        )
        print(f"{record_id}: {deltas_str} | weighted_delta="
              f"{final['weighted_delta']:.3f} | min_delta={final['min_delta']:.3f} "
              f"| {len(history) - 1}/{budget} muts", flush=True)

    detail_fields = ["id", "mutations_used", "mutation_budget", "objective",
                     "min_delta", "mean_delta", "weighted_delta",
                     "unweighted_predicted_rna_dna_ratio",
                     "weighted_predicted_rna_dna_ratio",
                     "predicted_rna_dna_ratio"]
    for name in args.models:
        detail_fields += [f"weight_{name}", f"original_{name}", f"final_{name}",
                          f"delta_{name}"]
    detail_fields.append("new_sequence")
    mb.write_tsv(output_dir / "optimized_sequences_detailed.tsv",
                 results, detail_fields)

    history_fields = ["id", "step", "objective", "min_delta", "mean_delta",
                      "weighted_delta", "weighted_prediction"]
    for name in args.models:
        history_fields += [f"weight_{name}", f"reg_{name}", f"delta_{name}"]
    history_fields += ["mutation", "position", "from", "to",
                       "mutations_used", "sequence"]
    mb.write_tsv(output_dir / "optimization_history.tsv",
                 all_history, history_fields)

    mb.write_tsv(
        output_dir / "submission.tsv",
        [{"id": row["id"], "new_sequence": row["new_sequence"],
          "predicted_rna_dna_ratio": row["predicted_rna_dna_ratio"]}
         for row in results],
        ["id", "new_sequence", "predicted_rna_dna_ratio"],
    )

    print(f"\nsaved to: {output_dir}")


if __name__ == "__main__":
    main()
