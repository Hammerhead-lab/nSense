# Handoff: Limits of Bias — Innate / Derived / Neutral Classifier

> For Claude Code. Read this whole file before writing code. Build only what is in **Build now**. Anything under **Later** is out of scope; do not scaffold it.

## 1. What this project is

A data mining course project (UVic CSC503). We classify words in sentences into three classes of bias, using embeddings from a pretrained language model and a random forest.

| Class | Meaning | Example |
|---|---|---|
| **innate** | The group is part of the word's meaning (dictionary sense) | *girls*, *queen*, *Muslim* |
| **derived** | The word is linked to a group through stereotype, not meaning | *soft* (in "Girls tend to be more soft…"), *nurse* |
| **neutral** | No group information | *fish*, *chair* |

**Course goal (all we need now):** build the model, show it works on held-out words, and compare it to a simple baseline.

## 2. The model

```
f(o) = c(e(o)) = class

o = (s, span)
    s    = a sentence
    span = character span of the word being classified in s
e = bert-base-uncased, final hidden layer
    → 768-d vector for the span (mean of its subword token vectors)
c = sklearn RandomForestClassifier
class ∈ {innate, derived, neutral}
```

## 3. Data: StereoSet (do NOT commit it to the repo)

- Load with `datasets`: `load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation", revision=<PINNED>)`.
- Resolve the current commit hash of the dataset repo once, write it into `config.yaml` as `dataset_revision`, and always load with it.
- 2,106 items. License CC BY-SA 4.0: cite Nadeem et al. 2021 in the README.
- Fields per item: `id`, `target`, `bias_type`, `context` (contains `BLANK`), `sentences` → each with `sentence`, `id`, `labels`, `gold_label`.
- `gold_label` codes: `0` = anti-stereotype, `1` = stereotype, `2` = unrelated. **Verify these against the dataset card before relying on them.**

### Mapping to our classes

| StereoSet piece | → Our class |
|---|---|
| `target` word, found in the sentence | innate |
| Fill word of the sentence with `gold_label = 1` | derived |
| Fill word of the sentence with `gold_label = 2` | neutral |
| Sentence with `gold_label = 0` | drop |
| Any item with `bias_type == "profession"` | drop entirely |

Axes kept: `gender`, `race` (call it "race/nationality" in outputs), `religion`.

### Extracting spans
- **Fill word:** align `sentence` against `context`; the text that replaced `BLANK` is the fill. It may be several words.
- **Target word:** case-insensitive search for `target` in the sentence. Exact match plus a simple plural/suffix rule (e.g. `girl`→`girls`, `Ethiopia`→`Ethiopian`). If not found, drop that innate row.
- **Count and report every drop** (by reason and axis). Write the counts to `results/data_report.json`.
- One row per labeled span, so each item yields up to 3 rows (innate from the stereotype sentence, derived, neutral).

## 4. Input mode (TEAM DECISION — implement both, behind one flag)

`config.yaml: input_mode: sentence | template`

- **`sentence` (Option B):** o = (StereoSet sentence, span). Natural context.
- **`template` (Option A):** take the unique words per class from the mapping above and place each into neutral templates (SEAT-style, May et al. 2019), e.g. `"This is {w}."`, `"{w} is here."`, `"It is {w}."`. The span is the inserted word. Grammar will sometimes be imperfect; that is acceptable — note it in `docs/DECISIONS.md`.

Everything downstream of the rows table must be identical for both modes. Default: `sentence`.

## 5. Evaluation

- **Grouping (non-negotiable):** `GroupKFold(n_splits=5)` with groups = the lowercased word. The same word must never appear in both train and test. Assert this in code.
- Report per fold and averaged: macro-F1, per-class precision/recall/F1 (`classification_report`), confusion matrix.
- Report **train score vs held-out score** side by side (overfitting check).
- Report counts: rows per class per axis, and **unique words per class** (innate is expected to be ~50 unique words — flag if far off).
- Permutation importance on held-out folds (`sklearn.inspection.permutation_importance`), saved as CSV.

## 6. Baselines (same splits, same metrics)

1. **Single-direction baseline:** for each axis, compute the centroid of innate vectors *from the training fold only*. Feature = cosine(vector, centroid of its axis). Fit a `DecisionTreeClassifier(max_depth=2)` on that one feature. This is the "bias is one direction" baseline the forest must beat.
2. **Part-of-speech baseline:** spaCy POS tag of the span's head word in its sentence, one-hot → `DecisionTreeClassifier`. Checks whether classes are just noun vs adjective.
3. **Frequency baseline:** `wordfreq.zipf_frequency(word, "en")` → `DecisionTreeClassifier(max_depth=2)`. Checks whether classes are just word frequency.

Output a single comparison table: model × macro-F1 (mean ± std over folds).

## 7. Project structure

```
limits-of-bias/
  config.yaml        # seed, model, layer, input_mode, axes, dataset_revision, paths
  requirements.txt   # pinned with ==
  Dockerfile
  README.md          # how to run, citation for StereoSet
  docs/
    HANDOFF.md       # this file
    DECISIONS.md     # log every judgement call (drop rules, templates, edge cases)
  src/
    data.py          # StereoSet → data/rows.csv (word, span, sentence, class, axis, item_id)
    embed.py         # rows.csv → data/embeddings.npy (+ data/embeddings_meta.json)
    train.py         # forest + GroupKFold → results/
    baselines.py     # the three baselines → results/
    utils.py         # seeding, config loading
  data/              # generated; gitignored
  results/           # generated; gitignored except a final summary
  notebooks/         # exploration and plots only; nothing the pipeline depends on
```

- Each stage reads the previous stage's file and writes its own. No stage imports another stage's internals.
- `embed.py` runs once; teammates without a GPU work from `embeddings.npy`.
- `embeddings_meta.json` records model name, layer, pooling, input_mode, dataset_revision, row count.

## 8. Reproducibility

- One `seed` in `config.yaml`. Set `random`, `numpy`, and `torch` seeds in `utils.py`; pass `random_state=seed` to every sklearn estimator that takes one.
- BERT in `eval()` mode, `torch.no_grad()`. Use a fast tokenizer with `return_offsets_mapping=True` to map character spans to tokens.
- Pin all package versions. Pin the dataset revision.
- Use CUDA if available, else CPU. Results must not depend on which.

## 9. Docker

- CPU-only image (`python:3.11-slim`, CPU build of torch). No GPU passthrough.
- Download the BERT model and the pinned dataset revision at **build time** so the container runs offline.
- Default command runs the full pipeline: `data → embed → train → baselines`.

## 10. Check-in points (stop and show the human)

1. After `data.py`: print rows per class × axis, unique words per class, and drop counts. **Stop for review** before embedding.
2. After `train.py` + `baselines.py`: print the comparison table and the train-vs-held-out scores.

## 11. Later — do NOT build now

PCA (only if train ≫ held-out), RoBERTa-base second model, whitening, layer sweep, clean subset by annotator agreement (5/5), mining forest split points as learned limits, wealth axis (CrowS-Pairs), HolisticBias innate terms, reweighted cosine similarity.

## 12. References

- Nadeem et al. 2021, StereoSet: https://arxiv.org/abs/2004.09456
- May et al. 2019, SEAT templates: https://arxiv.org/abs/1903.10561
- Blodgett et al. 2021, pitfalls in StereoSet (threat to validity): https://aclanthology.org/2021.acl-long.81
