# CHANGE-03: Held-out test set, label-granularity comparison, robustness study, baselines, committed results

> For Claude Code. Purpose: make the experiment match the course's Evaluation protocol (Milestone 4).
> The dataset and the encoder are **frozen**: do not change `data.py`'s class mapping, drop rules, row
> selection or grouping, and do not change `embed.py`. Anything not listed here is out of scope.

## 0. Before you start

- Branch from `main` (it includes Carson's fold fix, commit `4ceeab5`): `change-03`.
- Read `docs/HANDOFF.md`, `docs/DECISIONS.md`, and this file.
- The protocol in one line: **lock a test set first → every choice is made by 5-fold CV on the rest →
  each model's chosen configuration is scored on the test set exactly once.**
- The research questions this change serves:
  - **RQ1 (label granularity):** with the same vectors and the same learner, does splitting *bias*
    into *innate* and *derived* keep bias detection intact while adding a distinction a binary
    label cannot make?
  - **RQ2 (robustness):** does the RQ1 result hold across random forest configurations?
  - **Supporting:** is the random forest a reasonable learner, compared with conventional
    classifiers and the shortcut baselines?

## Change A: Dev/test split (`src/utils.py`, `config.yaml`)

- Add to `config.yaml`: `test_n_splits: 5` (the test set is one fold of 5, so about 20%).
- New function `dev_test_split(rows, cfg)`:
  - `StratifiedGroupKFold(n_splits=cfg["test_n_splits"], shuffle=True, random_state=seed)` with
    `y = class` (the three classes), `groups = group`. The **first** fold's held-out indices are the
    test set; everything else is the development (dev) set.
  - **Hard assertion:** no group appears in both dev and test.
  - Returns `dev_idx, test_idx` (indices into `rows`).
- A word-grouped split cannot be exactly 80/20. Report the **actual** ratio; do not adjust it.
- Write `results/split.csv` (row, item_id, word, group, axis, class, split) and
  `results/split_report.json`:
  - rows and groups per split
  - class counts and percentages per split
  - **class × axis counts per split**
  - the actual test share

## Change B: CV on the dev set only (`src/utils.py`)

- New function `dev_folds(rows, cfg)`: calls `dev_test_split`, then runs the existing `group_folds`
  (Carson's version, unchanged) on the **dev rows only**, and maps the fold indices back to row
  indices. Returns `dev_idx, test_idx, folds`.
- `train.py` and `baselines.py` both call `dev_folds`, so every model gets identical folds by
  construction. Keep the existing no-overlap assertion on every fold.
- **The test set is read in exactly one place: `src/final.py` (Change E).** `train.py` and
  `baselines.py` must never fit, select, scale or compute features using test rows. Add an assertion in
  each that no fold index is in `test_idx`.
- The single-direction centroid still comes from training-fold rows only (and, in Change E, from dev
  rows only).

**Check-in 1 (stop here).** Print `split_report.json` as tables, including class × axis per split, and
the assertion results. Flag any of these:
- the actual test share is outside 15–25%
- a class has under 10% of its rows in the test set
- a class × axis cell in the test set has under 20 rows

Wait for approval.

## Change C: Binary labelling, and the random forest study (`src/utils.py`, `src/train.py`, `config.yaml`)

**Binary labels.** Add to `config.yaml`:
```yaml
binary_map: {innate: bias, derived: bias, neutral: neutral}
```
Binary labels are derived **at training time** from `class`. `rows.csv` and `data.py` are unchanged:
same rows, same split, same folds.

Add these helpers to `utils.py`:
- `to_binary(labels)`: applies `binary_map`.
- `binary_metrics(y_true3, y_pred_binary)`: maps the true labels with `to_binary`, then computes
  macro-F1 over `{bias, neutral}`, per-class precision/recall/F1, and the 2×2 confusion matrix.

**Two random forests, one grid.** Add to `config.yaml`:
```yaml
rf_grid:
  n_estimators: [100, 300, 500]
  max_depth: [5, 10, 20, null]   # null = unlimited (the current setting)
```
Every other parameter stays at its scikit-learn default, with `random_state=seed, n_jobs=-1`.

For each of the 12 configurations, run 5-fold CV on the dev folds for both models:
- **3-class RF:** trained on `class`. Record:
  - 3-class macro-F1 (validation and train)
  - per-class F1
  - **collapsed bias-vs-neutral macro-F1**: its predictions mapped with `to_binary`, scored with
    `binary_metrics`
- **Binary RF:** trained on `to_binary(class)`. Record bias-vs-neutral macro-F1 (validation and train).

The **RQ1 quantity** for each configuration is the **gap**: 3-class collapsed bias-vs-neutral F1 minus
binary bias-vs-neutral F1, as a mean over folds with its std. The **innate-vs-derived evidence** is the
3-class model's per-class F1 for innate and derived. The binary model cannot produce these.

Write `results/rf_sweep.csv`, one row per configuration, with all of the above (mean and std). No
permutation importance during the sweep.

**Selection**, each model on its own native metric:
- 3-class RF: highest mean validation 3-class macro-F1.
- Binary RF: highest mean validation bias-vs-neutral macro-F1.
- Tie-break (within 0.001): fewer trees, then shallower.

Record both chosen configurations in `results/train_report.json` under `best_params`.

For the chosen 3-class configuration only, keep today's outputs computed on the dev folds:
per-fold metrics, `importances.csv` (permutation importance, as before), and `oof_predictions.csv`.

**Figures.** Use the Agg backend. Size each for one column of an ACM two-column page (about
3.3 × 2.2 in). Axis labels and a legend, no title (the caption goes in LaTeX). Save as PNG and PDF;
save the PDF with `metadata={"CreationDate": None}` so reruns are byte-identical.
1. `results/rf_sweep.{png,pdf}` (RQ2, sensitivity): x = `n_estimators`, y = mean CV 3-class macro-F1,
   one line per `max_depth`, error bars = std. Train macro-F1 as dashed lines in the same colour as
   their depth.
2. `results/label_gap.{png,pdf}` (RQ1 across settings): x = the 12 configurations, ordered by depth
   and then trees, with readable tick labels such as `d5·100`. Plot two lines, mean CV
   bias-vs-neutral F1 for the binary RF and for the collapsed 3-class RF, with error bars = std.

Add `matplotlib` to `requirements.txt` at a pinned version compatible with `numpy==1.26.4`. Log the
version in DECISIONS.

Remove the final "fit on all rows" step (`forest.joblib`). The final fit moves to Change E.

## Change D: Conventional baselines on the BERT vectors (`src/baselines.py`, `config.yaml`)

These are supporting rows, three-class only. Add two models on the same span embeddings
(`embeddings.npy`) and the same dev folds:
1. **Logistic regression:** `Pipeline([StandardScaler(), LogisticRegression(C=C, max_iter=5000,
   random_state=seed)])` (multinomial, the default `lbfgs` solver). Grid: `C ∈ {0.1, 1, 10}`.
2. **k-nearest neighbours:** `Pipeline([StandardScaler(), KNeighborsClassifier(n_neighbors=k)])`
   (Euclidean distance, uniform weights). Grid: `k ∈ {1, 3, 5, 7, 9}`.

Put both grids in `config.yaml` (`lr_grid`, `knn_grid`). The scaler is fit inside the pipeline, so it
only sees training-fold rows. Select each model's best value by mean validation 3-class macro-F1
(tie-break: the simpler model, meaning smaller C or larger k). Write every grid point to
`results/baseline_sweeps.csv`. If LR raises convergence warnings, report which C values did, and do
not change the grid.

The existing four baselines keep their fixed configurations (they are shortcut probes, not tuned
competitors), except **context-only**, which now uses the 3-class forest's **chosen** configuration
from `train_report.json` (`best_params`). Decision 32 already says it uses "the forest's exact
settings".

`comparison.csv` keeps its current format: 3-class CV results on the dev folds, one row per model, now
including LR and k-NN at their chosen values. Add one column, `bias_vs_neutral_f1`, the collapsed
binary score for every 3-class model. Add one row, `random_forest_binary`, which fills only that
column.

## Change E: Final test evaluation (new `src/final.py`)

This is the only stage that reads `test_idx`. Fit each model once on **all dev rows** with its chosen
or fixed configuration, then predict the test set once.

- **RQ1 models** (each at its own chosen configuration):
  - 3-class RF: 3-class metrics (macro-F1, per-class precision/recall/F1, 3×3 confusion matrix) **and**
    collapsed bias-vs-neutral metrics (macro-F1, per-class, 2×2 confusion matrix)
  - binary RF: bias-vs-neutral metrics (macro-F1, per-class, 2×2 confusion matrix)
  - the gap, as in Change C
- **Supporting models** (LR, k-NN, single-direction, part of speech, frequency, context-only):
  3-class metrics and the collapsed bias-vs-neutral macro-F1.
- **AUC**, reported only and never used for selection:
  - 3-class models: macro one-vs-rest AUC, `roc_auc_score(y, proba, multi_class="ovr",
    average="macro", labels=CLASSES)`, with the `predict_proba` columns checked against `CLASSES`
  - binary RF: binary AUC with `bias` as the positive class
- Write `results/test_report.json` (full) and `results/test_comparison.csv` (model, labelling,
  3-class macro-F1, bias-vs-neutral macro-F1, AUC, per-class F1).
- Add `python src/final.py` to the end of the Dockerfile `CMD`.

## Change F: Committed results and the run command

- `final.py` writes `results/summary.md`, the one file a reader needs:
  1. How to reproduce: `docker compose up --build`. Seed 503. Model and dataset revisions from
     `config.yaml`.
  2. The split, including class × axis (from `split_report.json`).
  3. **RQ1:** a test-set table with the binary RF and the collapsed 3-class RF on bias-vs-neutral F1,
     the gap, and the 3-class RF's innate and derived F1. Link to `label_gap.png`.
  4. **RQ2:** the RF sweep table (both models), the chosen configurations, and links to
     `rf_sweep.png` and `label_gap.png`.
  5. **Supporting:** CV results on the dev set for every model (`comparison.csv`) and test results
     (`test_comparison.csv`), plus the 3-class confusion matrices for the RF and the best-scoring
     baseline.
  6. Device and thread count from `embeddings_meta.json` and `os.cpu_count()`.

  No timestamps anywhere in committed files, so two runs can be compared with `diff`.
- `.gitignore`: also un-ignore these files:
  - `results/rf_sweep.png`, `results/rf_sweep.pdf`
  - `results/label_gap.png`, `results/label_gap.pdf`
  - `results/rf_sweep.csv`, `results/comparison.csv`, `results/baseline_sweeps.csv`
  - `results/test_comparison.csv`, `results/split_report.json`
- `docs/README.md`: add the run command and a pointer to `results/summary.md`.

## Change G: Documentation

- **`docs/DECISIONS.md`:**
  - Correct decision 30. Folds are no longer `GroupKFold`. They are Carson's deterministic
    size-balanced group assignment (`4ceeab5`), now applied to the dev set only.
  - Add new decisions from 34 onward for every judgement call in this change, including why each RF
    is tuned on its own native metric.
- **`docs/LIMITATIONS.md`:** replace the Oct 29 numbers with the new dev-CV and test numbers. Keep the
  existing items. If any class × axis cell in the test set is under 20 rows, add one paragraph saying
  which cells are thin and that per-axis results are not reported for them.
- **`docs/WALKTHROUGH.md`:**
  - update the "why we split by group" paragraph to describe the dev/test split
  - add a short paragraph on the binary comparison (same rows, labels merged at training time)
  - add `final.py` to the file list
- Do not touch the LaTeX reports in `docs/`.

## Check-in 2 (stop here)

Run the full pipeline in Docker **twice** on the same machine (`docker compose up --build`, then again),
and `diff` the committed `results/` files between the two runs. Print:

- `split_report.json` as tables (including class × axis)
- `rf_sweep.csv` as a table, and both chosen configurations
- **RQ1 on the test set:** binary RF vs collapsed 3-class RF on bias-vs-neutral F1, the gap, and the
  3-class RF's innate and derived F1
- the LR and k-NN grid results and their chosen values
- `comparison.csv` (dev CV) and `test_comparison.csv` (test) as tables
- the 3-class RF's train vs validation macro-F1 at its chosen configuration
- the test-set confusion matrices: 3-class RF (3×3 and collapsed 2×2), binary RF (2×2), best baseline
- the `diff` result (expected: no differences)
- wall-clock time for each stage

Then stop and list every file changed.
