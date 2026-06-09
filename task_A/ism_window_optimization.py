import argparse
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task_B"))
import multi_model_beam_search as mb
from ism_attribution import run_ism

BASES = mb.BASES
REGION_SIZE = 6
MAX_SPAN = 20
MAX_COST_INCREASE = 3
MAX_ITERATIONS = 50
BATCH_SIZE = 16
MIN_IMPROVEMENT = 1e-6
MODEL_NAMES = ["domi", "wika", "ola"]

def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def mutation_count(original, candidate):
    return sum(old != new for old, new in zip(original, candidate))

def mutation_description(original, candidate):
    return ", ".join(
        f"{position}:{old}>{new}"
        for position, (old, new) in enumerate(zip(original, candidate), start=1)
        if old != new
    )

def select_regions(ism_frame):
    positions = (
        ism_frame.sort_values("mean_delta", ascending=False)
        .groupby("full_position", as_index=False)
        .first()
    )
    anchor = int(positions.loc[positions["mean_delta"].idxmax(), "full_position"])
    best_mutation = None
    best_compact = None
    for start in positions["full_position"]:
        window = positions[
            positions["full_position"].between(start, start + MAX_SPAN)
        ]
        selected = window.nlargest(REGION_SIZE, "mean_delta")
        if len(selected) != REGION_SIZE:
            continue
        
        region = sorted(selected["full_position"].astype(int).tolist())
        score = float(selected["mean_delta"].sum())
        
        if best_compact is None or score > best_compact[0]:
            best_compact = (score, region)
            
        if anchor in region:
            if best_mutation is None or score > best_mutation[0]:
                best_mutation = (score, region)
    return {
        "best_mutation": best_mutation[1] if best_mutation else [],
        "best_compact": best_compact[1] if best_compact else [],
    }

def enumerate_window(sequence, positions):
    zero_based = [position - 1 for position in positions]
    seq_bytes = bytearray(sequence, "ascii")
    for combination in itertools.product(BASES, repeat=len(positions)):
        for index, base in zip(zero_based, combination):
            seq_bytes[index] = ord(base)
        yield seq_bytes.decode("ascii")

def run_window_bruteforce(
    scorers,
    original,
    current,
    original_scores,
    current_objective,
    positions,
    budget,
):
    current_cost = mutation_count(original, current)
    cost_limit = min(budget, current_cost + MAX_COST_INCREASE)
    variants = [
        (seq, cost) for seq in enumerate_window(current, positions)
        if (cost := mutation_count(original, seq)) <= cost_limit
    ]
    if not variants:
        return pd.DataFrame()

    sequences, costs = zip(*variants)
    
    scores = {
        scorer.name: np.asarray(scorer.predict(sequences), dtype=float)
        for scorer in scorers
    }
    deltas = {
        name: values - original_scores[name] for name, values in scores.items()
    }
    delta_values = np.vstack(list(deltas.values()))
    
    minimum = delta_values.min(axis=0)
    mean = delta_values.mean(axis=0)
    
    cost_increases = np.array(costs) - current_cost
    improvements = mean - current_objective
    
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiencies = np.where(
            cost_increases > 0, 
            improvements / cost_increases, 
            np.where(improvements > MIN_IMPROVEMENT, np.inf, -np.inf)
        )

    frame = pd.DataFrame({
        "mutations": [mutation_description(original, seq) for seq in sequences],
        "mutation_cost": costs,
        "cost_increase": cost_increases,
        "objective": mean,
        "objective_improvement": improvements,
        "gain_per_new_mutation": efficiencies,
        "min_delta": minimum,
        "mean_delta": mean,
        "sequence": sequences,
    })
    
    for name in scores:
        frame[f"score_{name}"] = scores[name]
        frame[f"delta_{name}"] = deltas[name]

    return frame.sort_values(
        [
            "gain_per_new_mutation",
            "objective_improvement",
            "objective",
            "min_delta",
        ],
        ascending=False,
    )

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("--record", required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()

def main():
    args = parse_args()
    original = mb.read_fasta(args.fasta)[args.record]
    current = original
    budget = args.budget or mb.DEFAULT_BUDGETS[args.record]
    output_dir = args.output_dir or Path(__file__).parent / "outputs" / args.record
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    scorers = [mb.SCORER_REGISTRY[name][0]( mb.SCORER_REGISTRY[name][1], device, BATCH_SIZE)
        for name in MODEL_NAMES
    ]
    for scorer in scorers:
        if hasattr(scorer, "lgb_reg"):
            scorer.lgb_reg.set_params(n_jobs=1)
            
    original_scores = {
        scorer.name: float(scorer.predict([original])[0])
        for scorer in scorers
    }
    current_objective = 0.0
    
    initial_state = {
        "iteration": 0,
        "region_selection": "",
        "positions": "",
        "mutations": "",
        "mutation_cost": 0,
        "cost_increase": 0,
        "objective": 0.0,
        "objective_improvement": 0.0,
        "gain_per_new_mutation": 0.0,
        "min_delta": 0.0,
        "mean_delta": 0.0,
        "sequence": original,
    }
    for name, score in original_scores.items():
        initial_state[f"score_{name}"] = score
        initial_state[f"delta_{name}"] = 0.0

    history = [initial_state]

    print(f"device: {device}")
    print(f"models: {MODEL_NAMES}")
    print("objective: mean")

    for iteration in range(1, MAX_ITERATIONS + 1):
        ism_frame, _ = run_ism(scorers, current)
        regions = select_regions(ism_frame)
        frames = []
        tested_regions = set()
        for name, positions in regions.items():
            region = tuple(positions)
            if not positions or region in tested_regions:
                continue
            tested_regions.add(region)
            frame = run_window_bruteforce(
                scorers,
                original,
                current,
                original_scores,
                current_objective,
                positions,
                budget,
            )
            if not frame.empty:
                frame.insert(0, "region_selection", name)
                frames.append(frame)

        if not frames:
            break

        candidates = pd.concat(frames, ignore_index=True).sort_values(
            [
                "gain_per_new_mutation",
                "objective_improvement",
                "objective",
                "min_delta",
            ],
            ascending=False,
        )
        best = candidates.iloc[0]
        selection = best["region_selection"]
        positions = regions[selection]
        improvement = float(best["objective"]) - current_objective
        
        print(
            f"iteration {iteration}: region={selection}, "
            f"positions={positions}, "
            f"cost={int(best['mutation_cost'])}/{budget}, "
            f"improvement={improvement:+.4f}"
        )
        
        if improvement <= MIN_IMPROVEMENT:
            break
            
        current = str(best["sequence"])
        current_objective = float(best["objective"])
        
        new_record = best.to_dict()
        new_record.update({
            "iteration": iteration,
            "region_selection": selection,
            "positions": ",".join(map(str, positions))
        })
        history.append(new_record)
        
        if int(best["mutation_cost"]) >= budget:
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
