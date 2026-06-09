"""In-silico saturation mutagenesis (ISM)

Example
-------
    python task_B/ism_attribution.py \
        --models domi wika ola --records seq_4_no_active
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import multi_model_beam_search as mb  # noqa: E402

BASES = mb.BASES
CORE_START = mb.CORE_START
CORE_END = mb.CORE_END


def saturation_variants(sequence, allowed_positions=None):
    variants, meta = [], []
    for position in range(CORE_START, CORE_END):
        if allowed_positions is not None and position not in allowed_positions:
            continue
        current = sequence[position]
        for alt in BASES:
            if alt == current:
                continue
            variants.append(sequence[:position] + alt + sequence[position + 1:])
            meta.append((position, current, alt))
    return variants, meta


def run_ism(scorers, sequence, allowed_positions=None):
    variants, meta = saturation_variants(sequence, allowed_positions)
    base_scores = {s.name: float(s.predict([sequence])[0]) for s in scorers}
    deltas = {}
    for scorer in scorers:
        scores = np.asarray(scorer.predict(variants), dtype=float)
        deltas[scorer.name] = scores - base_scores[scorer.name]

    names = [s.name for s in scorers]
    delta_stack = np.vstack([deltas[name] for name in names])
    rows = []
    for index, (position, current, alt) in enumerate(meta):
        row = {
            "full_position": position + 1,
            "core_position": position - CORE_START + 1,
            "current_base": current,
            "alt_base": alt,
        }
        for name in names:
            row[f"delta_{name}"] = float(deltas[name][index])
        row["min_delta"] = float(delta_stack[:, index].min())
        row["mean_delta"] = float(delta_stack[:, index].mean())
        rows.append(row)
    return pd.DataFrame(rows), base_scores


def compare_with_gradient(ism_df, gradient_path):
    """Spearman correlation + top-20 overlap between gradient and ISM best-gain
    per position, for Domi (gradient attribution is Domi-only)."""
    grad = pd.read_csv(gradient_path)

    def grad_best_gain(row):
        current = row["current_base"]
        base = float(row[f"gradient_estimated_score_if_{current}"])
        return max(
            float(row[f"gradient_estimated_score_if_{b}"]) - base
            for b in BASES if b != current
        )

    grad["best_gain_grad"] = grad.apply(grad_best_gain, axis=1)
    grad_pos = grad.groupby("full_position")["best_gain_grad"].max()
    ism_pos = ism_df.groupby("full_position")["delta_domi"].max().rename("best_gain_ism")

    merged = pd.concat([grad_pos, ism_pos], axis=1).dropna()
    spearman = merged.corr(method="spearman").iloc[0, 1]
    top_grad = set(merged["best_gain_grad"].nlargest(20).index)
    top_ism = set(merged["best_gain_ism"].nlargest(20).index)
    overlap = len(top_grad & top_ism)
    return spearman, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["domi", "wika", "ola"],
                        choices=list(mb.SCORER_REGISTRY))
    parser.add_argument("--records", nargs="+", default=None,
                        help="Subset of record ids; default = all in --fasta.")
    parser.add_argument("--prefilter-top", type=int, default=0,
                        help="If >0, only run ISM on the top-K gradient positions "
                             "(gradient -> ism -> beam). 0 = scan whole core.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--fasta", nargs="+", type=Path,
        default=[mb.ROOT / "task_materials" / "subtaskA.fa",
                 mb.ROOT / "task_materials" / "subtaskB.fa"],
    )
    parser.add_argument(
        "--gradient-dir", type=Path,
        default=mb.TASK2_DIR / "gradient_attribution" / "outputs",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=mb.TASK2_DIR / "ism_scanning" / "outputs",
    )
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | models: {args.models}")

    scorers = [
        mb.SCORER_REGISTRY[name][0](mb.SCORER_REGISTRY[name][1], device, args.batch_size)
        for name in args.models
    ]

    records = mb.load_records(args.fasta)
    targets = args.records or list(records)

    for record_id in targets:
        if record_id not in records:
            print(f"skip {record_id}: not found")
            continue
        sequence = records[record_id]

        allowed_positions = None
        if args.prefilter_top and args.prefilter_top > 0:
            grad_path = args.gradient_dir / f"{record_id}_attribution.csv"
            if not grad_path.exists():
                print(f"skip prefilter for {record_id}: {grad_path} missing")
            else:
                grad_rows = sorted(mb._gradient_position_gains(grad_path), reverse=True)
                allowed_positions = {
                    position for _, position in grad_rows[:args.prefilter_top]
                }

        ism_df, base_scores = run_ism(scorers, sequence, allowed_positions)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{record_id}_ism.csv"
        ism_df.to_csv(out_path, index=False)

        base_str = ", ".join(f"{n} {v:+.3f}" for n, v in base_scores.items())
        print(f"\n{record_id}  baseline: {base_str}")

        # Top positions by robust consensus (best achievable min_delta per pos).
        best_per_pos = (
            ism_df.groupby(["full_position", "current_base"])["min_delta"]
            .max().reset_index().sort_values("min_delta", ascending=False)
        )
        print("  top consensus positions (full_position: current -> best min_delta):")
        for _, r in best_per_pos.head(8).iterrows():
            print(f"    {int(r['full_position']):>3} {r['current_base']}  "
                  f"min_delta={r['min_delta']:+.3f}")

        # gradient-vs-ISM validation (Domi only).
        grad_path = args.gradient_dir / f"{record_id}_attribution.csv"
        if "domi" in args.models and grad_path.exists():
            spearman, overlap = compare_with_gradient(ism_df, grad_path)
            print(f"  gradient vs ISM (Domi): Spearman={spearman:.3f}, "
                  f"top-20 position overlap={overlap}/20")
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
