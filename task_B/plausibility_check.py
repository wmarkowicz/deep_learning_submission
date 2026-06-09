"""Biological-plausibility analysis for engineered DNA sequences

This script contains two families of checks:

1. Sequence complexity / low-complexity red flags
     GC%, longest homopolymer run, longest tandem repeat (period 1-3),
     Shannon entropy (mono- and di-nucleotide), and k-mer vocabulary usage.
   A sequence that won high model scores via a GC/AT repeat will light up here.

2. Transcription-factor motif content
     Scans both strands for a curated set of canonical motifs (IUPAC
     consensus) and reports how many were gained vs the original -- evidence
     that activity gains came from recreating real regulatory grammar rather
     than from junk.

Example
-------
    python plausibility_check.py \
        --run domi_wika_ola_min_ism
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE_START = 15
CORE_END = 215  # exclusive; mutations only ever happen inside [15, 215)

IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
    "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]",
    "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
}

# Curated canonical motifs (consensus, IUPAC). Not a full JASPAR scan -- a
# defensible starter set of well-known core-promoter / enhancer elements.
MOTIFS = {
    "TATA_box": "TATAWAW",
    "Initiator": "YYANWYY",
    "GC_box_Sp1": "GGGGCGG",
    "CAAT_box": "CCAAT",
    "E_box_bHLH": "CANNTG",
    "CRE": "TGACGTCA",
    "AP1": "TGASTCA",
    "NFkB": "GGGRNWYYCC",
    "ETS": "MGGAAGT",
    "GATA": "WGATAR",
    "Forkhead": "TGTTTAC",
    "Octamer": "ATGCAAAT",
}

# Heuristic thresholds for red flags.
HOMOPOLYMER_FLAG = 8       # run of one base this long or longer
TANDEM_REPEAT_FLAG = 12    # tandem repeat (period 1-3) this long or longer
MONO_ENTROPY_FLAG = 1.80   # bits, out of a max of 2.0
GC_LOW_FLAG, GC_HIGH_FLAG = 25.0, 75.0


def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def core(seq):
    return seq[CORE_START:CORE_END]


def gc_content(seq):
    if not seq:
        return 0.0
    return 100.0 * sum(b in "GC" for b in seq) / len(seq)


def longest_homopolymer(seq):
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def longest_tandem_repeat(seq):
    """Longest stretch that is a tandem repeat of period 1, 2 or 3 (catches
    homopolymers, GC/AT dinucleotide repeats, and trinucleotide repeats)."""
    best = 0
    for period in (1, 2, 3):
        run = period
        for i in range(period, len(seq)):
            if seq[i] == seq[i - period]:
                run += 1
                best = max(best, run)
            else:
                run = period
    return best


def shannon_entropy(seq, k):
    if len(seq) < k:
        return 0.0
    counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def kmer_vocabulary_usage(seq, k):
    """Fraction of distinct overlapping k-mers actually used, relative to the
    number that could possibly appear in a sequence of this length."""
    if len(seq) < k:
        return 0.0
    windows = len(seq) - k + 1
    distinct = len(set(seq[i:i + k] for i in range(windows)))
    return distinct / min(4 ** k, windows)


def count_motif(seq, pattern_regex):
    """Overlapping count on a single strand."""
    return len(re.findall(f"(?=({pattern_regex}))", seq))


def motif_counts(seq):
    counts = {}
    rc = reverse_complement(seq)
    for name, consensus in MOTIFS.items():
        regex = "".join(IUPAC[b] for b in consensus)
        counts[name] = count_motif(seq, regex) + count_motif(rc, regex)
    return counts


def complexity_profile(full_seq):
    c = core(full_seq)
    return {
        "gc_pct": gc_content(c),
        "longest_homopolymer": longest_homopolymer(c),
        "longest_tandem_repeat": longest_tandem_repeat(c),
        "entropy_mono_bits": shannon_entropy(c, 1),
        "entropy_di_bits": shannon_entropy(c, 2),
        "kmer3_usage": kmer_vocabulary_usage(c, 3),
    }


def flags_for(profile):
    flags = []
    if profile["longest_homopolymer"] >= HOMOPOLYMER_FLAG:
        flags.append(f"homopolymer x{profile['longest_homopolymer']}")
    if profile["longest_tandem_repeat"] >= TANDEM_REPEAT_FLAG:
        flags.append(f"tandem-repeat x{profile['longest_tandem_repeat']}")
    if profile["entropy_mono_bits"] < MONO_ENTROPY_FLAG:
        flags.append(f"low entropy {profile['entropy_mono_bits']:.2f}b")
    if not (GC_LOW_FLAG <= profile["gc_pct"] <= GC_HIGH_FLAG):
        flags.append(f"GC {profile['gc_pct']:.0f}%")
    return flags


def reference_stats(dataset_path, sample):
    """Complexity baseline from real *active* enhancers (is_active == 1)."""
    df = pd.read_csv(dataset_path, sep="\t")
    active = df[df["is_active"].astype(str) == "1"]["sequence"].astype(str)
    if len(active) > sample:
        active = active.sample(sample, random_state=0)
    profiles = [complexity_profile(s.upper()) for s in active]
    frame = pd.DataFrame(profiles)
    return frame.mean(), frame.quantile(0.05), frame.quantile(0.95)


def load_designs(run_dir, fasta_paths):
    sub = pd.read_csv(run_dir / "submission.tsv", sep="\t")
    originals = {}
    for path in fasta_paths:
        text = Path(path).read_text().strip()
        for entry in text.split(">"):
            if entry.strip():
                lines = entry.splitlines()
                originals[lines[0].strip()] = "".join(
                    x.strip() for x in lines[1:] if x.strip()
                ).upper()
    return sub, originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="domi_wika_ola_min_ism",
                        help="Output sub-folder under sequence_optimization/outputs/")
    parser.add_argument(
        "--fasta", nargs="+", type=Path,
        default=[ROOT / "task_materials" / "subtaskA.fa",
                 ROOT / "task_materials" / "subtaskB.fa"],
    )
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "task_materials" / "dataset_task1.tsv")
    parser.add_argument("--reference-sample", type=int, default=3000)
    args = parser.parse_args()

    run_dir = ROOT / "task_B" / "outputs" / args.run
    sub, originals = load_designs(run_dir, args.fasta)

    ref_mean = ref_lo = ref_hi = None
    if args.dataset.exists():
        ref_mean, ref_lo, ref_hi = reference_stats(args.dataset, args.reference_sample)
        print(f"reference (real active enhancers, n<={args.reference_sample}): "
              f"GC {ref_mean['gc_pct']:.1f}% "
              f"[{ref_lo['gc_pct']:.0f}-{ref_hi['gc_pct']:.0f}], "
              f"mono-entropy {ref_mean['entropy_mono_bits']:.2f}b, "
              f"max homopolymer {ref_mean['longest_homopolymer']:.1f}, "
              f"max tandem-repeat {ref_mean['longest_tandem_repeat']:.1f}")

    summary_rows = []
    motif_rows = []
    for _, row in sub.iterrows():
        sid = row["id"]
        if sid not in originals:
            continue
        orig, new = originals[sid], str(row["new_sequence"]).upper()
        po, pn = complexity_profile(orig), complexity_profile(new)
        mo, mn = motif_counts(core(orig)), motif_counts(core(new))

        print(f"\n=== {sid}  (predicted_rna_dna_ratio = "
              f"{row.get('predicted_rna_dna_ratio', float('nan')):.2f}) ===")
        print(f"  GC%             {po['gc_pct']:5.1f} -> {pn['gc_pct']:5.1f}")
        print(f"  max homopolymer {po['longest_homopolymer']:5d} -> "
              f"{pn['longest_homopolymer']:5d}")
        print(f"  max tandem rpt  {po['longest_tandem_repeat']:5d} -> "
              f"{pn['longest_tandem_repeat']:5d}")
        print(f"  entropy mono(b) {po['entropy_mono_bits']:5.2f} -> "
              f"{pn['entropy_mono_bits']:5.2f}")
        print(f"  3-mer usage     {po['kmer3_usage']:5.2f} -> {pn['kmer3_usage']:5.2f}")
        flags = flags_for(pn)
        print(f"  RED FLAGS: {', '.join(flags) if flags else 'none'}")

        gained = {m: mn[m] - mo[m] for m in MOTIFS if mn[m] - mo[m] != 0}
        if gained:
            print("  motif change (gained/lost vs original):")
            for m, d in sorted(gained.items(), key=lambda x: -x[1]):
                print(f"     {m:12} {mo[m]} -> {mn[m]}  ({d:+d})")
        else:
            print("  motif change: none")

        srow = {"id": sid, "predicted_rna_dna_ratio":
                row.get("predicted_rna_dna_ratio", float("nan"))}
        for key in po:
            srow[f"orig_{key}"] = po[key]
            srow[f"new_{key}"] = pn[key]
        srow["red_flags"] = "; ".join(flags) if flags else ""
        summary_rows.append(srow)
        for m in MOTIFS:
            motif_rows.append({"id": sid, "motif": m,
                               "orig": mo[m], "new": mn[m], "delta": mn[m] - mo[m]})

    out_dir = run_dir / "plausibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "complexity_summary.tsv",
                                      sep="\t", index=False)
    pd.DataFrame(motif_rows).to_csv(out_dir / "motif_changes.tsv",
                                    sep="\t", index=False)
    print(f"\nsaved: {out_dir}")

    if ref_mean is not None:
        print("\nplausibility verdict (designs vs real active-enhancer range):")
        for srow in summary_rows:
            ok_gc = ref_lo["gc_pct"] <= srow["new_gc_pct"] <= ref_hi["gc_pct"]
            ok_ent = srow["new_entropy_mono_bits"] >= MONO_ENTROPY_FLAG
            ok_rep = srow["new_longest_tandem_repeat"] < TANDEM_REPEAT_FLAG
            verdict = "PLAUSIBLE" if (ok_gc and ok_ent and ok_rep) else "CHECK"
            print(f"  {srow['id']:16} {verdict}  "
                  f"(GC {'ok' if ok_gc else 'out'}, "
                  f"entropy {'ok' if ok_ent else 'low'}, "
                  f"repeats {'ok' if ok_rep else 'high'})")


if __name__ == "__main__":
    main()