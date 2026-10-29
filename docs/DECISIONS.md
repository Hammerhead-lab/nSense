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

## Template mode

13. **Templates**: `This is {w}.`, `{w} is here.`, `It is {w}.`, `That is {w}.`, `Here is {w}.`,
    `There is {w}.`. The word is inserted lowercased (the model is uncased). Grammar is sometimes off
    (*"This is poor."* is fine, *"This is mothers."* is not). The handoff says this is acceptable.
14. **One label per word in template mode.** Each template sentence is the same for every occurrence of a
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

## Open questions (check-in 1)

- **OPEN: Group innate rows by target, not surface word?** The 49 targets give 71 unique innate words,
  because of plurals and suffixes (*mother/mothers*, *ethiopia/ethiopian/ethiopians*). Grouping by
  lowercased word can put *mother* in train and *mothers* in test. That is near-duplicate leakage, and it
  helps the forest most on the innate class.
- **OPEN: Innate words that appear as derived fills.** *muslim* (15), *african* (3), *russian* (2),
  *arab*, *ethiopia*, *male* (1 each) occur as the stereotype fill, so they are labeled derived. By the
  class definitions they are innate. Options: keep as derived (follow the mapping), relabel as innate,
  or drop.
