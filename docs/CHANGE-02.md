# CHANGE-02: Build the model stages, final grouping fix, documentation

> For Claude Code. The dataset is **frozen** after this change: do not alter the class mapping, drop rules, or row selection beyond Change A. Anything not listed here is out of scope.

## 0. Before you start

- Branch from `main` after CHANGE-01 is merged: `change-02`.
- Read `docs/HANDOFF.md`, `docs/DECISIONS.md`, and this file.

## Change A: Merge groups that share a surface word (resolves the open *ethiopian* item)

- After `group` is assigned (decision 17), merge any groups that share a lowercased surface `word` (union-find). The merged group's name is the alphabetically first member.
- Expected on the pinned revision: `ethiopia` + `ethiopian` merge; nothing else. Report every merge in `results/data_report.json` under `merged_groups`.
- Add an assertion: every lowercased surface word maps to exactly one group.
- Log it in `docs/DECISIONS.md` and mark the open item resolved.

## Change B: Report class concentration (document only, no reweighting)

Add to `data_report.json`, per class: the top-10 groups by row count, and their combined share of that class's rows.

## Change C: `src/embed.py`

Per HANDOFF §2, §7, §8:
- `bert-base-uncased` at the pinned revision (decision 3), `eval()`, `torch.no_grad()`, final hidden layer.
- **Span vector:** mean of the token vectors whose character offsets overlap `[span_start, span_end)` (fast tokenizer, `return_offsets_mapping=True`).
- **Context vector (for the baseline in Change E):** mean of all token vectors in the sentence **excluding** the span tokens and the special tokens `[CLS]`/`[SEP]`.
- **Alignment check:** for every row, decode the span tokens and assert they equal the lowercased `word` (ignoring `##` and spaces). Fail loudly on any mismatch.
- Save `data/embeddings.npy` (span vectors), `data/context_embeddings.npy`, and `data/embeddings_meta.json` (model, revision, layer, pooling, dataset revision, row count, device used).
- CUDA if available, else CPU.

## Change D: `src/train.py`

- `RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)`.
- `GroupKFold(n_splits=5)` on `group`. **Hard assertion:** no group appears in both train and test in any fold.
- Per fold and mean ± std: macro-F1, per-class precision/recall/F1, confusion matrix, **train macro-F1 next to held-out macro-F1**.
- Permutation importance on each held-out fold (`n_repeats=5`, `random_state=seed`, scoring `f1_macro`). Save the mean across folds to `results/importances.csv`.
- Save per-row out-of-fold predictions to `results/oof_predictions.csv` (row id, true class, predicted class, fold), for later review.
- Fit one final forest on all rows; save it with `joblib` to `results/forest.joblib`.

## Change E: `src/baselines.py`

Same folds, same metrics as `train.py`:
1. **Single-direction** (HANDOFF §6.1)
2. **Part of speech** (HANDOFF §6.2)
3. **Frequency** (HANDOFF §6.3)
4. **Context-only (new).** The same random forest settings, trained on `context_embeddings.npy` (the sentence without the word). Purpose: neutral words always come from StereoSet's unrelated (nonsense) sentences, so the forest might detect the odd sentence rather than the word. If context-only scores high on **neutral**, that confound is real. This is a check to report, not something to fix.

Write `results/comparison.csv`: model × macro-F1 (mean ± std) × per-class F1.

## Change F: Documentation

**`docs/WALKTHROUGH.md`**: plain language, written for a teammate who has not seen the code, at most about 2 pages.
- The model in one line: c(e(o)).
- Trace **one real StereoSet item** end to end: raw item → three rows → vectors → prediction.
- What each `src/` file does, its input file, its output file.
- Why we split by group (one paragraph, with the *mother/mothers* example).
- What each baseline tests, in one line each.
- How to rerun (local and Docker), where results land.
- "If something breaks": the most likely failure points (alignment check, group assertion, dependency versions) and where to look.

**`docs/LIMITATIONS.md`**: a draft paragraph per item, factual, using the actual numbers:
- Neutral context confound (cite the context-only baseline result)
- Cross-axis rows held out (24), and why the class is relative to the axis
- Class concentration (the top-10 shares from Change B)
- StereoSet validity issues (Blodgett et al. 2021)
- One encoder (BERT-base-uncased), one layer; results may not hold for other models

## Check-in 2 (stop here)

Run `data → embed → train → baselines`. Print:
- `results/comparison.csv` as a table
- train vs held-out macro-F1 for the forest
- per-class F1 for the forest and the context-only baseline, side by side
- the alignment-check and group-assertion results
- `merged_groups` and the class-concentration figures

Then stop and list every file changed.
