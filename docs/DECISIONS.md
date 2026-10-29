# Decisions

Every judgement call made while building, newest at the bottom. Open questions are marked **OPEN**.

## Data (src/data.py)

1. **Dataset revision.** `McGill-NLP/stereoset` pinned to commit `bf6e7ce50491784d094fb7afe60a70ecccb89035`
   (last modified 2024-01-23). The repo is plain parquet, so no loading script runs.
2. **gold_label codes verified.** The dataset card's ClassLabel names are
   `0 = anti-stereotype, 1 = stereotype, 2 = unrelated`, as the handoff says. `data.py` asserts
   this at load time.
3. **Model revision pinned too.** `bert-base-uncased` pinned to `86b5e0934494bd15c9632b12f734a8a67f723594`.
   The handoff only asks to pin the dataset, but an unpinned model can change results in the same way.
4. **Rows schema.** `item_id, sentence_id, axis, class, word, group, span_start, span_end, sentence`.
   The span is stored as two integer columns rather than one `span` field. `group` is `word.lower()`
   and is the GroupKFold key, so every later stage uses the same grouping.
5. **Axis label.** `race` is written as `race/nationality` in rows.csv and in every output.
6. **Fill extraction.** Split `context` at `BLANK` into prefix and suffix. The fill is what lies between
   them in the sentence, matched case-insensitively, then stripped of edge whitespace and punctuation.
   A fuzzy fallback (common prefix/suffix) exists for edited sentences. On the pinned revision every
   kept sentence aligned exactly, and every fill is a single word.
7. **Contexts with two BLANKs are dropped** (1 item: *"brahmin is very BLANK and is strict about BLANK"*).
   This drops its derived row, its neutral row and its innate row.
8. **Target search.** Case-insensitive, at word boundaries, with suffixes `s, es, n, ns, an, ans, ese`
   and `y → ies`. Multi-word targets allow any whitespace between words. Matches that overlap the fill
   span are ignored. The first remaining match is used.
9. **Innate rows come from the stereotype sentence only**: one per item, as the handoff says.
10. **Missed targets are not fixed by hand** (4 items). Two are dataset typos (*"gentlement"*). In one,
    *Sierra Leon* is written *Sierra Leone*. In the last, *Arab* only appears inside *Arabia*, which is
    correctly not a match.
11. **Target typos kept as they are**: *eriteria*, *norweigan*, *columbian*. They are the words in the
    sentences, so BERT sees them as written.
12. **Sentence mode keeps words that appear in more than one class** (59 words, e.g. *poor*: derived 86,
    neutral 1). In sentence mode the context differs, so both labels can be correct. All rows of a word
    share one group, so this cannot leak across folds.

## Removed in CHANGE-01 (team chose Option B). Template mode

13. **Removed in CHANGE-01 (team chose Option B).** **Templates**: `This is {w}.`, `{w} is here.`, `It is {w}.`, `That is {w}.`, `Here is {w}.`,
    `There is {w}.`. The word is inserted lowercased (the model is uncased). Grammar is sometimes off
    (*"This is poor."* is fine, *"This is mothers."* is not). The handoff says this is acceptable.
14. **Removed in CHANGE-01 (team chose Option B).** **One label per word in template mode.** Each template sentence is the same for every occurrence of a
    word, so a word with several labels would give identical vectors with different labels. Rule:
    majority class wins and ties are dropped (24 words on the pinned revision, all 1-vs-1
    derived/neutral). The axis is the majority axis within the winning class. `item_id` lists every
    source item for that word, separated by `;`.

## Environment

15. **Full dependency lock.** The first build pulled a pyarrow that needs NumPy 2. `requirements.txt` is
    now the full `pip freeze` of the built image, plus the spaCy model wheel by URL. torch `2.5.1+cpu` is
    installed from the CPU wheel index in the Dockerfile.
16. **The container runs as uid 1000, not root**, so files written to the bind-mounted `data/` and
    `results/` belong to the host user.

## CHANGE-01 (Option B only, grouping, cross-axis rows)

17. **Grouping key (resolves OPEN item 1).** `group` is now the item's `target`, lowercased and
    whitespace-normalized, for innate rows, and the spaCy lemma of the fill word, lowercased, for derived
    and neutral rows. `word` stays the surface form. On the pinned revision: 49 innate groups (one per
    kept target, from 71 surface words), derived 600 words → 575 groups, neutral 761 words → 720 groups.
18. **Lemmas are taken out of context.** The lowercased fill word is lemmatized alone, not inside its
    sentence (`en_core_web_sm` 3.7.1, full pipeline). In context the same surface form can get different
    lemmas (*left* → *leave* or *left*) and so land in two groups, which would put one word in train and
    test. Out of context each surface word gets exactly one lemma, so groups only ever merge words.
    Accepted quirks: *better* → *well*, *boxing* → *box*.
19. **Cross-axis rows are held out (resolves OPEN item 2).** The class is relative to the axis: *muslim*
    is innate to religion but is a stereotype fill for a race/nationality item. The classifier has no axis
    input, so these rows would give contradictory labels. A derived or neutral row whose fill matches any
    kept target under the decision-8 rules is removed from `rows.csv` and written to
    `data/cross_axis_probe.csv` (rows schema plus `source_class` and `matched_target`). It is not embedded
    or evaluated yet. On the pinned revision: 24 derived rows, 0 neutral. *muslim* 15 (14
    race/nationality, 1 religion), *african* 3, *russian* 2, *muslims* 1, *arab* 1, *ethiopia* 1,
    *male* 1. That is one more than CHANGE-01 expected: *muslims* matches *muslim* through the `s` suffix.
20. **Cross-axis matching details.** Kept targets are the 49 normalized targets of every gender, race
    and religion item, including items whose rows were dropped. A match on the item's own axis also
    counts (*male* in a gender item, *muslim* in a religion item), as CHANGE-01 says "any" target.
    `matched_target` is the first match in alphabetical order (no fill matched more than one). Removed
    rows are counted in `drops` as `derived_cross_axis` / `neutral_cross_axis`. The output path is
    `paths.cross_axis_probe` in `config.yaml`.
21. **Report keys.** `group` is no longer the word, so `unique_words_by_class(_axis)` and
    `words_in_multiple_classes` are now computed on the lowercased `word` to keep their meaning. New keys:
    `kept_targets`, `unique_groups_by_class(_axis)`, `cross_axis_probe_rows`, `cross_axis_probe_by_class`,
    `cross_axis_probe_by_word`. `input_mode` is removed.
22. **`click==8.1.8` added to `requirements.txt`.** In the built image `import spacy` failed with
    `No module named 'click'`. The pinned `typer` no longer installs it. With the pin, `pip check` is clean.
23. **README location.** The README is `docs/README.md`; there is no root `README.md`. Template mode and
    `input_mode` were removed from `docs/README.md`.
24. **Stages not in the repo.** `src/embed.py`, `src/train.py` and `src/baselines.py` do not exist on
    this branch. There was no `embeddings_meta.json` code to change, and no GroupKFold assertion to check.
    Recheck both when those stages are added.

## CHANGE-02 (model stages, final grouping fix)

25. **Groups that share a surface word are merged (resolves the *ethiopian* open item).** After decision 17,
    a union-find joins any groups that share a lowercased `word`. The merged group is named after its
    alphabetically first member. `data.py` asserts that every lowercased word maps to exactly one group.
    On the pinned revision only `ethiopia` + `ethiopian` → `ethiopia` (53 innate rows), so there are 48
    innate groups for 49 kept targets. Merges are listed under `merged_groups` in `data_report.json`.
26. **Class concentration is reported, not corrected.** `class_concentration` in `data_report.json`
    gives the top-10 groups per class and their share of the class's rows: derived 21.2% (*poor* alone
    is 86 rows), innate 24.9%, neutral 12.2%. No reweighting.
27. **Fill spans are widened to whole words.** One gender item has the context *"The grandfather
    BLANKed the grandchildren."*, so the fill text was only the stem (*protect*, *beard*) and the span
    ended inside a word. The `embed.py` alignment check failed on these 2 rows. The team chose to extend
    a fill span to the word boundaries, so `word` is now *protected* / *bearded*. Rows, classes and
    selection are unchanged. Group: *protect* (unchanged); *bearded* (was *beard*, as spaCy keeps the
    adjective). They show as `exact_widened_to_word: 2` under `stats.fill_alignment`.
28. **Span tokens.** A token belongs to the span if its character offsets overlap
    `[span_start, span_end)` and it is not `[CLS]`/`[SEP]`/padding. The alignment check joins the span
    tokens with `##` removed and compares them with the lowercased `word` without spaces. All 3,857 rows
    pass.
29. **Context vector.** The mean of every non-special token outside the span (CHANGE-02). `embed.py`
    asserts that each row has at least one such token. Saved to `data/context_embeddings.npy`
    (`paths.context_embeddings`).
30. **Shared folds and metrics live in `utils.py`.** `group_folds` (GroupKFold plus the hard no-overlap
    assertion) and `evaluate` (per-fold and mean ± std metrics) are used by both `train.py` and
    `baselines.py`, so the folds and metrics are the same by construction. Stages still do not import
    each other. GroupKFold is deterministic, so no seed is involved.
31. **Permutation importance runs with `n_jobs=-1`** across features, on each held-out fold. With an
    integer `random_state` the result does not depend on the number of workers. `importances.csv` holds
    the mean across folds and the std across folds per feature, sorted by mean.
32. **Baseline details.** Single direction: the per-axis innate centroid comes from training rows only;
    the feature is the cosine to the centroid of the row's own axis. Part of speech: `doc.char_span(...,
    alignment_mode="expand").root.pos_` in the full sentence, one-hot over all tags seen, then an
    unlimited-depth tree (HANDOFF gives no depth). Frequency: `zipf_frequency(word, "en")` of the surface
    word. Context-only: the forest's exact settings on the context vectors. All trees use
    `random_state=seed`.
33. **`comparison.csv`** has one row per model: held-out macro-F1 mean and std, held-out per-class F1
    (mean over folds) and train macro-F1. `baselines.py` reads the forest row from
    `results/train_report.json`. Full per-fold reports are in `train_report.json` and
    `baselines_report.json`.

## Open questions (check-in 1)

- **Resolved in CHANGE-01 (decision 17).** Group innate rows by target, not surface word? The 49 targets give 71 unique innate words,
  because of plurals and suffixes (*mother/mothers*, *ethiopia/ethiopian/ethiopians*). Grouping by
  lowercased word can put *mother* in train and *mothers* in test. That is near-duplicate leakage, and it
  helps the forest most on the innate class.
- **Resolved in CHANGE-01 (decision 19).** Innate words that appear as derived fills. *muslim* (15), *african* (3), *russian* (2),
  *arab*, *ethiopia*, *male* (1 each) occur as the stereotype fill, so they are labeled derived. By the
  class definitions they are innate. Options: keep as derived (follow the mapping), relabel as innate,
  or drop.

## Open questions (CHANGE-01 check-in)

- **Resolved in CHANGE-02 (decision 25).** *ethiopian* sits in two innate groups. *Ethiopia* and *Ethiopian* are both kept targets. The
  surface word *ethiopian* is innate in group `ethiopia` (3 rows, suffix match) and in group `ethiopian`
  (17 rows), so GroupKFold can put the same word in train and test. It is the only surface word in more
  than one group. Options: merge the two targets into one group, or leave as is.
