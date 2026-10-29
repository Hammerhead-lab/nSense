# CHANGE-01: Option B only, grouping fix, cross-axis rows

> For Claude Code. Apply these three changes to the existing build. Change nothing else. Do not refactor or restyle unrelated code.

## 0. Before you start

- If the git working tree has uncommitted changes, stop and ask the human to commit first.
- `docs/HANDOFF.md` has been replaced with a version where the team chose **Option B only** (o = (s, i), StereoSet sentences as-is). Read it again before editing.
- Make these changes on a branch named `change-01`.

## Change 1: Remove template mode

The team chose Option B. Template mode was built from an earlier handoff and must be removed.

- Delete all template-mode code paths, template constants, and any tests for them.
- Remove `input_mode` from `config.yaml`, from `embeddings_meta.json` generation, and from the README.
- Rows are always produced the sentence-mode way.
- In `docs/DECISIONS.md`, do **not** delete decisions 13–14. Prefix the "Template mode" heading and both entries with **Removed in CHANGE-01 (team chose Option B).**

## Change 2: Grouping key (resolves OPEN item 1)

Grouping by lowercased surface word leaks near-duplicates across folds (*mother* in train, *mothers* in test), which inflates the innate class. Change the `group` column in `rows.csv`:

| Row class | `group` value |
|---|---|
| innate | the item's `target` field, lowercased, whitespace-normalized |
| derived, neutral | spaCy lemma of the fill word, lowercased (use the already-installed spaCy model) |

- Keep `word` as the surface form. Only `group` changes.
- `train.py` and `baselines.py` already group by `group`; confirm they need no change.
- Add a `DECISIONS.md` entry recording this rule and marking OPEN item 1 resolved.

## Change 3: Cross-axis rows (resolves OPEN item 2)

Some innate words appear as stereotype fills (e.g. *muslim* as the fill in an *Arab* item). Such a word is innate to its own axis (religion) but derived relative to the item's axis (race/nationality). The classifier has no axis input, so these rows give contradictory training labels.

- **Definition:** a derived or neutral row is cross-axis if its fill word matches **any** StereoSet target from the kept axes, using the same matching rules as target search (decision 8: case-insensitive, word boundaries, same suffix list).
- Remove these rows from `rows.csv`.
- Write them to `data/cross_axis_probe.csv`: same schema as `rows.csv`, plus `source_class` (the class the mapping gave it) and `matched_target`.
- Expected: about 23 derived rows (*muslim* 15, *african* 3, *russian* 2, *arab* 1, *ethiopia* 1, *male* 1). Report the actual count, and any neutral rows caught.
- Do **not** embed or evaluate the probe set now. It is listed under Later.
- Add a `DECISIONS.md` entry: the class is relative to the axis; these rows are held out of training as a cross-axis probe set; OPEN item 2 resolved.

## Report updates

Add to `results/data_report.json`:
- unique **groups** per class per axis
- cross-axis probe row count, broken down by word

## Acceptance checks (all must pass)

1. `grep -ri template src/ config.yaml README.md` returns nothing. `docs/` is exempt.
2. In `rows.csv`, the number of unique innate groups equals the number of kept targets (expected around 49).
3. No `rows.csv` row with class derived or neutral has a fill word that matches any target under the decision-8 rules.
4. Every word in `data/cross_axis_probe.csv` matches a target under the decision-8 rules.
5. The existing GroupKFold no-overlap assertion is unchanged and still enforced on `group`.

## Check-in (stop here)

Rerun `data.py` only. Print:
- rows per class × axis
- unique groups per class × axis
- drop counts by reason
- cross-axis probe count by word
- results of acceptance checks 1–5

Then stop. Do not run `embed.py`, `train.py`, or `baselines.py` until the human approves.

Finally, list every file you changed.
