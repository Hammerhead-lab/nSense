# CHANGE-04: Bias-vs-neutral AUC-ROC for RQ1

> For Claude Code. Purpose: compare the binary RF and the 3-class RF on the **same** threshold-free
> metric, so RQ1 can say whether the 3-class model is *better at ranking* bias above neutral or only
> *more cautious at its default cutoff*.
> Everything from CHANGE-03 is **frozen**: the split, the folds, the grids, the chosen configurations,
> the selection rules and the dataset. This change only **adds measurements**. No model choice may
> depend on them. Anything not listed here is out of scope.

## 0. Before you start

- Branch from `main` (it includes CHANGE-03, merge `e18a56b`): `change-04`.
- Read `docs/HANDOFF.md`, `docs/DECISIONS.md`, and this file.

## Change A: Bias scores (`src/utils.py`)

Add `bias_score(proba, classes)`. It takes `predict_proba` output and the model's `classes_`:
- **3-class model:** P(innate) + P(derived), with the columns looked up by name in `classes_`, never
  by position.
- **Binary model:** P(bias), looked up by name.

The true label for the AUC is `to_binary(class) == "bias"`.

## Change B: Cross-validated AUC for the two RQ1 models (`src/train.py`)

For the **chosen** 3-class configuration and the **chosen** binary configuration only (from
`best_params`; do not re-run the sweep), on the same 5 dev folds:
- Per fold, record the bias-vs-neutral AUC of each model, and the **paired difference** (3-class minus
  binary) on that fold.
- Write mean and std over folds to `results/train_report.json` under `rq1_auc_cv`, and print them.

If refitting is needed to get probabilities, refit with the same configuration and `random_state`.
The fold-level F1 numbers already reported must not change. Check this.

## Change C: Test-set AUC and the ROC figure (`src/final.py`)

On the test set, using the models already fit in `final.py`:

1. **Bias-vs-neutral AUC** for the binary RF and the collapsed 3-class RF.
2. **Collapsed bias-vs-neutral AUC for every other 3-class model** too (LR, k-NN, single-direction,
   part of speech, frequency, context-only). Add it as a new column, `bvn_auc`, in
   `results/test_comparison.csv`. Keep the existing 3-class macro AUC column as it is.
3. **Default operating points.** For each of the two RQ1 models, report bias precision, bias recall,
   neutral recall, the false-alarm count (neutral predicted as bias) and the miss count (bias predicted
   as neutral), computed from their actual predictions. The 3-class point uses the collapsed
   predictions.
4. **Matched-threshold check.** Take the 3-class model's false-alarm rate (FPR) at its default
   operating point. Find the binary RF's bias recall at that same FPR on its test ROC curve (use the
   highest TPR whose FPR does not exceed it). Report both recalls side by side. This answers whether
   the binary model could match the 3-class model just by moving its cutoff.
5. **Figure `results/roc_bvn.{png,pdf}`:**
   - test-set ROC curves for the binary RF and the collapsed 3-class RF
   - the AUC of each in the legend
   - each model's default operating point marked on its curve
   - a dashed chance diagonal

   Use the same style rules as CHANGE-03: one column wide (about 3.3 × 2.2 in), the Agg backend, no
   title, axis labels "False positive rate (neutral flagged as bias)" and "True positive rate (bias
   recall)", and the PDF saved with `metadata={"CreationDate": None}`.

Write all of the above to `results/test_report.json` under `rq1_auc_test`.

## Change D: Summary, gitignore, docs

- `results/summary.md`: add an **RQ1: ranking vs cutoff** subsection under RQ1 with:
  - the CV AUC table (both models, plus the paired difference, mean ± std)
  - the test AUC for both models
  - the two default operating points (precision, recalls, false alarms, misses)
  - the matched-threshold check
  - a link to `roc_bvn.png`
- `.gitignore`: un-ignore `results/roc_bvn.png` and `results/roc_bvn.pdf`.
- `docs/DECISIONS.md`: new decisions from 52 onward. Include how the bias score is built, and that these
  metrics are reported only and never used for selection.
- `docs/LIMITATIONS.md`: one sentence. The test-set operating-point comparison is a single split, and
  the CV AUC difference is what decides the claim.
- Do not touch the LaTeX reports.

## Check-in (stop here)

Run the full pipeline in Docker **twice** on the same machine and `diff` the committed `results/` files.
Print:
- the CV AUC table, including the paired difference
- the test AUCs, the two operating points and the matched-threshold check
- the new `bvn_auc` column of `test_comparison.csv`
- confirmation that every CHANGE-03 number in `summary.md` is unchanged (diff it against the
  version on `main`)
- the `diff` result between the two runs (expected: no differences)

Then stop and list every file changed. Do not commit until approved.
