# Task A: DNA Restoration Challenge

This directory contains the final Task A solution based on iterative
multi-model ISM and window brute force.

## Method

Each iteration:

1. evaluates every single substitution in the 200 nt enhancer core,
2. selects one region around the best mutation and one compact region with the
   strongest total ISM signal,
3. evaluates all nucleotide combinations at six selected positions,
4. ranks candidates by mean Domi/Wika/Ola improvement per new mutation,
5. accepts the better candidate and repeats until the mutation budget is used.

The optimization uses the mean improvement across the Domi, Wika, and Ola
models. The final search configuration is fixed in the script.

Model loading and prediction are shared with `task_B/multi_model_beam_search.py`.

The fixed budgets are 40 mutations for Seq1, 16 for Seq2, and 20 for Seq3.
Seq1 was rerun from `task_materials/subtaskA_flank_fixed.fa`, where the
shared 5' flank is restored before core optimization.

## Files

```text
task_A/
├── ism_window_optimization.py
├── run_all.sh
├── requirements.txt
├── outputs/
│   ├── seq1_broken_best_regions/
│   ├── seq1_broken_best_regions_flank_fixed/
│   ├── seq_2_broken_best_regions/
│   └── seq_3_broken_best_regions/
└── submission/
    ├── seq1.fa
    ├── seq2.fa
    └── seq3.fa
```

`submission/` contains the three sequences intended for submission. Seq1 uses
35 optimized mutations from the flank-fixed run.

## Environment

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r task_A/requirements.txt
```

## Run

```bash
bash task_A/run_all.sh
```

The optimizer writes `history.csv` under
`task_A/outputs/<record>_best_regions/`. Its last row contains the final result.
For the final Seq1 flank-fixed run:

```bash
python task_A/ism_window_optimization.py task_materials/subtaskA_flank_fixed.fa \
  --record seq1_broken \
  --budget 35 \
  --output-dir task_A/outputs/seq1_broken_best_regions_flank_fixed
```

## Results

| Sequence | Mutations | Mean | Domi | Wika | Ola |
|---|---:|---:|---:|---:|---:|
| Seq1, shared 5' flank restored | 35 (+5 flank restoration) | 2.761 | 2.084 | 2.977 | 3.222 |
| Seq1, original 5' flank | 40 | 2.863 | 2.142 | 3.056 | 3.393 |
| Seq2 | 16 | 2.313 | 2.141 | 2.731 | 2.066 |
| Seq3 | 20 | 1.982 | 1.566 | 1.681 | 2.699 |

For Seq1 we report both variants because the 5' flank differs between the
provided broken sequence and the shared flank used by the other Task A records.
The original-flank variant scores higher under the surrogate models, especially
Wika and Ola, which use sequence features affected by flank nucleotides. The
shared-flank variant is more conservative with respect to adapter consistency
and avoids optimizing signal from a likely non-enhancer flank region.