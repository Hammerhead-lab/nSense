"""Stage 1: StereoSet (intrasentence) -> data/rows.csv.

One row per labeled span:
  innate  = the item's target word, found in its stereotype sentence
  derived = the fill word of the stereotype sentence      (gold_label 1)
  neutral = the fill word of the unrelated sentence       (gold_label 2)
Anti-stereotype sentences (gold_label 0) and the profession axis are dropped.
Derived/neutral rows whose fill matches any kept target are cross-axis: they go to
data/cross_axis_probe.csv instead of rows.csv.
Every drop is counted by reason and axis in results/data_report.json.
"""
import re
from collections import Counter, defaultdict

import pandas as pd
import spacy
from datasets import load_dataset

from utils import load_config, path, set_seed, write_json

# gold_label codes, checked against the dataset card's ClassLabel names at load time.
ANTI, STEREO, UNRELATED = 0, 1, 2
EXPECTED_GOLD_NAMES = ["anti-stereotype", "stereotype", "unrelated"]

# Simple plural/suffix rule for finding the target in a sentence (girl->girls, Ethiopia->Ethiopian).
TARGET_SUFFIXES = ["", "s", "es", "n", "ns", "an", "ans", "ese"]
EDGE_CHARS = " \t\n\"'`.,;:!?()[]{}"

COLUMNS = ["item_id", "sentence_id", "axis", "class", "word", "group",
           "span_start", "span_end", "sentence"]
PROBE_COLUMNS = COLUMNS + ["source_class", "matched_target"]

SPACY_MODEL = "en_core_web_sm"


def load_items(cfg):
    ds = load_dataset(cfg["dataset_name"], cfg["dataset_config"],
                      split=cfg["dataset_split"], revision=cfg["dataset_revision"])
    gold_names = ds.features["sentences"].feature["gold_label"].names
    assert gold_names == EXPECTED_GOLD_NAMES, f"gold_label codes changed: {gold_names}"
    return ds


def trim(text, start, end):
    """Shrink [start, end) so it does not begin or end with whitespace/punctuation."""
    while start < end and text[start] in EDGE_CHARS:
        start += 1
    while end > start and text[end - 1] in EDGE_CHARS:
        end -= 1
    return start, end


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def extract_fill(context, sentence):
    """Return (start, end, how) of the text that replaced BLANK, or (None, None, drop_reason)."""
    if context.count("BLANK") != 1:
        return None, None, "context_blank_count"
    pre, suf = context.split("BLANK")
    c, s = context.lower(), sentence.lower()
    pre_l, suf_l = pre.lower(), suf.lower()

    if len(s) >= len(pre) + len(suf) and s.startswith(pre_l) and s.endswith(suf_l):
        start, end, how = len(pre), len(sentence) - len(suf), "exact"
    else:
        # The annotator changed text outside the blank (often punctuation or spacing).
        # Anchor on the longest common prefix/suffix, clipped to the blank's position.
        p = min(common_prefix(c, s), len(pre))
        q = min(common_prefix(c[::-1], s[::-1]), len(suf))
        if len(s) - q <= p:
            return None, None, "fill_misaligned"
        start, end, how = p, len(sentence) - q, "fuzzy"

    start, end = trim(sentence, start, end)
    if start >= end:
        return None, None, "fill_empty"
    # A context like "BLANKed" leaves only the stem as fill; widen the span to the whole word.
    s0, e0 = start, end
    while start > 0 and sentence[start - 1].isalnum():
        start -= 1
    while end < len(sentence) and sentence[end].isalnum():
        end += 1
    if (start, end) != (s0, e0):
        how += "_widened_to_word"
    return start, end, how


def normalize_target(target):
    """Lowercased, whitespace-normalized target: the innate group key."""
    return " ".join(target.lower().split())


def target_pattern(target):
    """Decision-8 rule: case-insensitive, at word boundaries, with the suffix list and y -> ies."""
    t = re.escape(target.strip().lower()).replace(r"\ ", r"\s+")
    alts = [t + "(?:" + "|".join(TARGET_SUFFIXES) + ")"]
    if target.lower().endswith("y"):
        alts.append(t[:-1] + "ies")  # mommy -> mommies
    return re.compile(r"(?<![a-z])(?:" + "|".join(alts) + r")(?![a-z])")


def find_target(target, sentence, avoid):
    """Case-insensitive search for the target (plus a simple suffix) outside the span `avoid`.

    Returns (start, end, how) or (None, None, drop_reason).
    """
    hits = list(target_pattern(target).finditer(sentence.lower()))
    if not hits:
        return None, None, "target_not_found"
    outside = [m for m in hits if m.end() <= avoid[0] or m.start() >= avoid[1]]
    if not outside:
        return None, None, "target_only_in_fill"
    m = outside[0]
    how = "exact" if m.group(0) == target.strip().lower() else "suffix"
    return m.start(), m.end(), how


def lemmatizer():
    """spaCy lemma of a lowercased word, taken out of context so a surface form always gets one group."""
    nlp = spacy.load(SPACY_MODEL)
    cache = {}

    def lemma(word):
        w = word.lower()
        if w not in cache:
            cache[w] = " ".join(t.lemma_.lower() for t in nlp(w))
        return cache[w]
    return lemma


def make_row(item, sent_id, axis, cls, sentence, start, end, group):
    word = sentence[start:end]
    return {"item_id": item["id"], "sentence_id": sent_id, "axis": axis, "class": cls,
            "word": word, "group": group, "span_start": start, "span_end": end,
            "sentence": sentence}


def match_any_target(word, patterns):
    """First kept target (sorted) whose decision-8 pattern matches the word, else None."""
    for target, pattern in patterns:
        if pattern.search(word.lower()):
            return target
    return None


def sentence_rows(ds, cfg):
    axes = set(cfg["axes"])
    lemma = lemmatizer()
    labels = cfg.get("axis_labels", {})
    rows = []
    drops = defaultdict(Counter)       # reason -> axis -> count
    stats = defaultdict(Counter)       # what -> how -> count
    items_by_type = Counter(ds["bias_type"])

    for item in ds:
        raw_axis = item["bias_type"]
        if raw_axis == "profession":
            drops["item_profession"][raw_axis] += 1
            continue
        if raw_axis not in axes:
            drops["item_axis_not_kept"][raw_axis] += 1
            continue
        axis = labels.get(raw_axis, raw_axis)

        sents = item["sentences"]
        golds = Counter(sents["gold_label"])
        if any(golds[g] != 1 for g in (ANTI, STEREO, UNRELATED)):
            stats["item_gold_label_counts_not_1_1_1"][axis] += 1

        innate_done = False
        for text, sid, gold in zip(sents["sentence"], sents["id"], sents["gold_label"]):
            if gold == ANTI:
                drops["sentence_anti_stereotype"][axis] += 1
                continue
            cls = "derived" if gold == STEREO else "neutral"

            start, end, how = extract_fill(item["context"], text)
            if start is None:
                drops[f"{cls}_{how}"][axis] += 1
                if gold == STEREO:
                    drops["innate_no_usable_stereotype_sentence"][axis] += 1
                continue
            stats["fill_alignment"][how] += 1
            rows.append(make_row(item, sid, axis, cls, text, start, end,
                                 lemma(text[start:end])))

            if gold == STEREO and not innate_done:
                t_start, t_end, t_how = find_target(item["target"], text, (start, end))
                if t_start is None:
                    drops[f"innate_{t_how}"][axis] += 1
                else:
                    stats["innate_match"][t_how] += 1
                    rows.append(make_row(item, sid, axis, "innate", text, t_start, t_end,
                                         normalize_target(item["target"])))
                innate_done = True

    # Cross-axis: a derived/neutral fill that matches any kept target (decision 8 rules)
    # is held out of training as a probe set.
    targets = sorted({normalize_target(item["target"]) for item in ds
                      if item["bias_type"] in axes})
    patterns = [(t, target_pattern(t)) for t in targets]
    df = pd.DataFrame(rows, columns=COLUMNS)
    matched = [None if c == "innate" else match_any_target(w, patterns)
               for c, w in zip(df["class"], df["word"])]
    df["matched_target"] = matched
    is_probe = df["matched_target"].notna()
    probe = df[is_probe].assign(source_class=df.loc[is_probe, "class"])[PROBE_COLUMNS]
    for cls, axis in zip(probe["class"], probe["axis"]):
        drops[f"{cls}_cross_axis"][axis] += 1
    df = df[~is_probe][COLUMNS]

    report = {
        "items_total": len(ds),
        "items_by_bias_type": dict(items_by_type),
        "kept_targets": targets,
        "drops": {r: dict(c) for r, c in sorted(drops.items())},
        "stats": {k: dict(c) for k, c in stats.items()},
    }
    return df.reset_index(drop=True), probe.reset_index(drop=True), report


def merge_shared_words(df):
    """Union-find: merge groups that share a lowercased surface word. Name = alphabetically first member."""
    parent = {g: g for g in df["group"].unique()}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for _, groups in df.groupby(df["word"].str.lower())["group"]:
        first, *rest = groups.unique()
        for g in rest:
            a, b = find(first), find(g)
            if a != b:
                parent[max(a, b)] = min(a, b)

    members = defaultdict(list)
    for g in sorted(parent):
        members[find(g)].append(g)
    merged = {root: ms for root, ms in members.items() if len(ms) > 1}
    df = df.assign(group=df["group"].map(find))

    groups_per_word = df.groupby(df["word"].str.lower())["group"].nunique()
    assert (groups_per_word == 1).all(), \
        f"words in more than one group: {groups_per_word[groups_per_word > 1].index.tolist()}"
    return df, merged


def concentration(df, k=10):
    """Per class: the top-k groups by row count and their combined share of the class's rows."""
    out = {}
    for cls, g in df.groupby("class"):
        top = g["group"].value_counts().head(k)
        out[cls] = {"top_groups": top.to_dict(), "top_share": round(top.sum() / len(g), 4)}
    return out


def summarize(df, probe):
    words = df["word"].str.lower()
    by_class_axis = pd.crosstab(df["class"], df["axis"], margins=True)
    unique = words.groupby(df["class"]).nunique()
    unique_axis = pd.crosstab(df.assign(w=words).drop_duplicates(["class", "axis", "w"])["class"],
                              df.assign(w=words).drop_duplicates(["class", "axis", "w"])["axis"])
    groups_axis = pd.crosstab(df.drop_duplicates(["class", "axis", "group"])["class"],
                              df.drop_duplicates(["class", "axis", "group"])["axis"])
    classes_per_word = df["class"].groupby(words).nunique()
    multi = df[words.isin(classes_per_word[classes_per_word > 1].index)]
    multi_words = {w: g["class"].value_counts().to_dict()
                   for w, g in multi.groupby(multi["word"].str.lower())}
    innate_words = sorted(words[df["class"] == "innate"].unique())
    probe_words = probe["word"].str.lower().value_counts()
    return {
        "rows_total": len(df),
        "rows_by_class_axis": {c: by_class_axis.loc[c].to_dict() for c in by_class_axis.index},
        "unique_words_by_class": unique.to_dict(),
        "unique_words_by_class_axis": {c: unique_axis.loc[c].to_dict() for c in unique_axis.index},
        "unique_groups_by_class": df.groupby("class")["group"].nunique().to_dict(),
        "unique_groups_by_class_axis": {c: groups_axis.loc[c].to_dict() for c in groups_axis.index},
        "words_in_multiple_classes": multi_words,
        "innate_words": innate_words,
        "cross_axis_probe_rows": len(probe),
        "cross_axis_probe_by_class": probe["source_class"].value_counts().to_dict(),
        "cross_axis_probe_by_word": probe_words.to_dict(),
        "class_concentration": concentration(df),
    }, by_class_axis, groups_axis


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    ds = load_items(cfg)
    df, probe, report = sentence_rows(ds, cfg)
    df, report["merged_groups"] = merge_shared_words(df)

    # Spans must point at the word they claim to.
    for d in (df, probe):
        assert all(s[a:b] == w for s, a, b, w in
                   zip(d["sentence"], d["span_start"], d["span_end"], d["word"]))

    summary, by_class_axis, groups_axis = summarize(df, probe)
    report = {"dataset_revision": cfg["dataset_revision"], **report, **summary}
    df.to_csv(path(cfg, "rows"), index=False)
    probe.to_csv(path(cfg, "cross_axis_probe"), index=False)
    write_json(report, path(cfg, "data_report"))

    print(f"rows = {len(df)}\n")
    print("Rows per class x axis:\n", by_class_axis, "\n", sep="")
    print("Unique groups per class x axis:\n", groups_axis, "\n", sep="")
    print("Unique groups per class:", summary["unique_groups_by_class"])
    print("Unique words per class:", summary["unique_words_by_class"])
    n_innate = summary["unique_words_by_class"].get("innate", 0)
    if not 25 <= n_innate <= 100:
        print(f"  WARNING: {n_innate} unique innate words; handoff expects ~50.")
    print(f"Words labeled with more than one class: {len(summary['words_in_multiple_classes'])}")
    print("Merged groups (shared surface word):", report["merged_groups"] or "none")
    print("Every lowercased word maps to exactly one group: OK")
    print("\nClass concentration (top-10 groups):")
    for cls, c in summary["class_concentration"].items():
        print(f"  {cls}: share {c['top_share']:.1%}  {c['top_groups']}")
    print("\nDrops (reason: axis counts):")
    for reason, counts in report["drops"].items():
        print(f"  {reason}: {counts}  (total {sum(counts.values())})")
    print(f"\nCross-axis probe rows: {len(probe)}  by class: {summary['cross_axis_probe_by_class']}")
    for w, n in summary["cross_axis_probe_by_word"].items():
        print(f"  {w}: {n}")
    print("\nStats:")
    for k, v in report["stats"].items():
        print(f"  {k}: {v}")
    print(f"\nWrote {path(cfg, 'rows')}, {path(cfg, 'cross_axis_probe')} and {path(cfg, 'data_report')}")


if __name__ == "__main__":
    main()
