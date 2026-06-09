# Task B: Super-Enhancer Challenge

Task B is about redesigning one DNA sequence with naturally low regulatory activity.

- input: `task_materials/subtaskB.fa`
- record id: `seq_4_no_active`
- goal: maximize predicted `rna_dna_ratio`
- mutation budget: no formal limit in the assignment
- required approach: iterative, model-guided optimization rather than random trial and error
- biological constraint: the optimized sequence should stay plausible, not just exploit the models with simple GC/AT repeats

The assignment asks us to defend the design using model interpretation methods. This solution uses in-silico mutagenesis and consensus beam search across the three group models: `domi`, `wika`, and `ola`.

## Why This Approach

The final strategy is a weighted, ISM-prioritized beam search. It was chosen because Task B is a difficult sequence-design problem: the search space is enormous, there is no explicit mutation budget, and optimizing against a single model can easily produce artificial sequences that exploit one predictor rather than genuinely improving regulatory activity.

ISM is used first because it gives an interpretable and model-evaluated estimate of which positions are most useful to edit. Unlike gradient attribution, which is a local approximation and was available mainly through the Domi model, ISM directly tests single-base substitutions and records the predicted effect for all selected models. This makes the candidate-position prior more reliable and easier to defend in the report: the beam search is not exploring arbitrary mutations, but positions that already showed positive single-mutation evidence.

Beam search was selected instead of a purely greedy search because beneficial multi-mutation designs may require keeping several competing partial sequences alive. A greedy algorithm can commit too early to a mutation that looks best at one step but blocks a stronger combination later. Beam search keeps the top candidates at each step, which gives a better balance between exploration and tractability than exhaustive search. It also scales well for this task because the beam width and ISM top-position cutoff directly control the number of evaluated candidates, so the method can be expanded or restricted depending on the available compute budget.

The final objective is `weighted_mean_floor`. The weighted mean lets the optimizer give more influence to the strongest individual model (`domi=0.60`) while still using `wika` and `ola` as regularizers. The floor term prevents the search from sacrificing one model to improve another: candidates that fail to achieve the required per-model improvement are heavily penalized. This was important because the goal is not only to maximize the average prediction, but to produce a sequence that all three independently trained models consider improved.

Several parameter combinations were tested before selecting the final run, including different objectives (`min`, `mean`, `mean_floor`, and weighted variants), different model weights, beam widths, patience values, ISM top-position cutoffs, and mutation caps. The selected configuration gave the best practical trade-off in these experiments: it achieved a high final mean prediction while keeping improvements positive across all three models and avoiding obvious plausibility problems. For this reason, it was treated as the most reliable final approach rather than simply the most aggressive optimizer setting.

The practical cap of 80 mutations and the final plausibility checks were added to keep the optimized sequence biologically reasonable. Since the assignment does not impose a strict Task B mutation budget, this cap acts as a safeguard against runaway optimization and simple repeat-based model exploitation.

## Files

```text
task_B/
├── ism_attribution.py
├── multi_model_beam_search.py
├── weighted_multi_model_beam_search.py
├── plausibility_check.py
└── outputs/B_weighted_domi060_floor318_top120/
    ├── submission.tsv
    ├── optimized_sequences_detailed.tsv
    ├── optimization_history.tsv
    └── plausibility/
```

- `ism_attribution.py` runs in-silico saturation mutagenesis. It tests single-base substitutions in the mutable core and reports score changes for the selected models.
- `multi_model_beam_search.py` runs consensus beam search with unweighted objectives: `min`, `mean`, or `mean_floor`.
- `weighted_multi_model_beam_search.py` is the main final optimizer. It supports model weights, so the strongest model can get more influence while the others still regularize the design.
- `plausibility_check.py` checks whether the optimized sequence looks biologically plausible: GC%, homopolymers, tandem repeats, entropy, 3-mer usage, and simple TF motif changes.

## Environment

Run setup commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r task_1_solution_domi/requirements.txt joblib lightgbm
```

`lightgbm` is needed to load the saved `.pkl` models. The scripts usually find the project root automatically. If they are copied elsewhere, set:

```bash
export DLS_PROJECT_ROOT=/path/to/deep_learning_submission
```

## Main Command

This is the most important command for the final Task B run. It assumes the current working directory is `task_B/`.

```bash
python weighted_multi_model_beam_search.py \
  --models domi wika ola \
  --weights domi=0.60 wika=0.25 ola=0.15 \
  --objective weighted_mean_floor \
  --floor 3.18 \
  --prediction-report mean \
  --prior ism \
  --beam-width 8 \
  --patience 25 \
  --top-positions 120 \
  --batch-size 2048 \
  --fasta ../task_materials/subtaskB.fa \
  --budget-overrides seq_4_no_active=80 \
  --output-dir outputs/B_weighted_domi060_floor318_top120
```

Although Task B has no formal mutation limit, the script still uses `seq_4_no_active=80` as a runtime and plausibility control. In the saved final run, the search stopped after 61 mutations.

## Pipeline

### 1. Run ISM Attribution

The final command uses `--prior ism`, so it expects an ISM file with ranked mutation positions. Run this from the repository root:

```bash
python task_B/ism_attribution.py \
  --models domi wika ola \
  --records seq_4_no_active \
  --fasta task_materials/subtaskB.fa \
  --output-dir task_B/ism_scanning/outputs \
  --batch-size 1024
```

Expected output:

```text
task_B/ism_scanning/outputs/seq_4_no_active_ism.csv
```

Important columns:

- `full_position`, `current_base`, `alt_base` - tested mutation
- `delta_domi`, `delta_wika`, `delta_ola` - model-specific prediction changes
- `min_delta` - worst improvement across models
- `mean_delta` - average improvement across models

### 2. Run Weighted Beam Search

Run the main command from `task_B/`. The optimizer writes:

- `submission.tsv` - final deliverable with `id`, `new_sequence`, `predicted_rna_dna_ratio`
- `optimized_sequences_detailed.tsv` - final sequence plus per-model predictions and deltas
- `optimization_history.tsv` - step-by-step mutation history for optimization-curve plots

Saved final result for `seq_4_no_active`:

```text
predicted_rna_dna_ratio: 2.8389
mutations_used: 61 / 80
domi: -0.9990 -> 2.1814
wika: -0.8893 -> 2.9260
ola:  -0.2123 -> 3.4093
```

### 3. Check Biological Plausibility

After generating `submission.tsv`, run:

```bash
python task_B/plausibility_check.py \
  --run B_weighted_domi060_floor318_top120 \
  --fasta task_materials/subtaskB.fa
```

Outputs:

```text
task_B/outputs/B_weighted_domi060_floor318_top120/plausibility/
```

Current final-run summary:

```text
GC%: 42.5 -> 41.0
longest homopolymer: 4 -> 7
longest tandem repeat: 8 -> 7
mono entropy: 1.97 -> 1.95
red flags: none
```

## Key Parameters

- `--models domi wika ola` - models used for consensus scoring
- `--weights domi=0.60 wika=0.25 ola=0.15` - model weights in the final run
- `--objective weighted_mean_floor` - weighted objective with a per-model safety floor
- `--floor 3.18` - minimum desired improvement per model; candidates below it receive a strong penalty
- `--prior ism` - restricts beam-search positions to ISM-ranked candidates
- `--top-positions 120` - number of top ISM positions used by beam search
- `--beam-width 8` - number of candidates kept after each step
- `--patience 25` - number of non-improving steps tolerated before stopping
- `--batch-size 2048` - prediction batch size
- `--budget-overrides seq_4_no_active=80` - practical mutation cap
- `--prediction-report mean` - writes the unweighted mean model prediction to `submission.tsv`

## Deliverable

```text
task_B/outputs/B_weighted_domi060_floor318_top120/submission.tsv
```
