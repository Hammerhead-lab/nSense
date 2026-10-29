"""Stage 1: StereoSet (intrasentence) -> data/rows.csv.

One row per labeled span:
  innate  = the item's target word, found in its stereotype sentence
  derived = the fill word of the stereotype sentence      (gold_label 1)
  neutral = the fill word of the unrelated sentence       (gold_label 2)
Anti-stereotype sentences (gold_label 0) and the profession axis are dropped.
Every drop is counted by reason and axis in results/data_report.json.
"""
import re
from collections import Counter, defaultdict

import pandas as pd
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
    return start, end, how


def find_target(target, sentence, avoid):
    """Case-insensitive search for the target (plus a simple suffix) outside the span `avoid`.

    Returns (start, end, how) or (None, None, drop_reason).
    """
    t = re.escape(target.strip().lower()).replace(r"\ ", r"\s+")
    alts = [t + "(?:" + "|".join(TARGET_SUFFIXES) + ")"]
    if target.lower().endswith("y"):
        alts.append(t[:-1] + "ies")  # mommy -> mommies
    pattern = re.compile(r"(?<![a-z])(?:" + "|".join(alts) + r")(?![a-z])")

    hits = list(pattern.finditer(sentence.lower()))
    if not hits:
        return None, None, "target_not_found"
    outside = [m for m in hits if m.end() <= avoid[0] or m.start() >= avoid[1]]
    if not outside:
        return None, None, "target_only_in_fill"
    m = outside[0]
    how = "exact" if m.group(0) == target.strip().lower() else "suffix"
    return m.start(), m.end(), how


def make_row(item, sent_id, axis, cls, sentence, start, end):
    word = sentence[start:end]
    return {"item_id": item["id"], "sentence_id": sent_id, "axis": axis, "class": cls,
            "word": word, "group": word.lower(), "span_start": start, "span_end": end,
            "sentence": sentence}


def sentence_rows(ds, cfg):
    axes = set(cfg["axes"])
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
            rows.append(make_row(item, sid, axis, cls, text, start, end))

            if gold == STEREO and not innate_done:
                t_start, t_end, t_how = find_target(item["target"], text, (start, end))
                if t_start is None:
                    drops[f"innate_{t_how}"][axis] += 1
                else:
                    stats["innate_match"][t_how] += 1
                    rows.append(make_row(item, sid, axis, "innate", text, t_start, t_end))
                innate_done = True

    report = {
        "items_total": len(ds),
        "items_by_bias_type": dict(items_by_type),
        "drops": {r: dict(c) for r, c in sorted(drops.items())},
        "stats": {k: dict(c) for k, c in stats.items()},
    }
    return pd.DataFrame(rows, columns=COLUMNS), report


def resolve_word_labels(df):
    """One (class, axis) per unique word for template mode: majority vote, ties dropped."""
    kept, dropped = [], []
    for group, g in df.groupby("group", sort=True):
        cls_counts = g["class"].value_counts()
        if len(cls_counts) > 1 and cls_counts.iloc[0] == cls_counts.iloc[1]:
            dropped.append({"word": group, "classes": cls_counts.to_dict()})
            continue
        cls = cls_counts.index[0]
        axis = g.loc[g["class"] == cls, "axis"].value_counts().index[0]
        ids = ";".join(sorted(g.loc[g["class"] == cls, "item_id"].unique()))
        kept.append({"group": group, "class": cls, "axis": axis, "item_id": ids})
    return kept, dropped


def template_rows(df, cfg):
    words, tied = resolve_word_labels(df)
    rows = []
    for w in words:
        for k, tpl in enumerate(cfg["templates"]):
            start = tpl.index("{w}")
            sentence = tpl.replace("{w}", w["group"])
            end = start + len(w["group"])
            rows.append({"item_id": w["item_id"], "sentence_id": f"template:{k}",
                         "axis": w["axis"], "class": w["class"], "word": w["group"],
                         "group": w["group"], "span_start": start, "span_end": end,
                         "sentence": sentence})
    report = {"template_count": len(cfg["templates"]),
              "words_dropped_class_tie": tied}
    return pd.DataFrame(rows, columns=COLUMNS), report


def summarize(df, src_df):
    by_class_axis = pd.crosstab(df["class"], df["axis"], margins=True)
    unique = df.groupby("class")["group"].nunique()
    unique_axis = pd.crosstab(df.drop_duplicates(["class", "axis", "group"])["class"],
                              df.drop_duplicates(["class", "axis", "group"])["axis"])
    classes_per_word = src_df.groupby("group")["class"].nunique()
    multi = src_df[src_df["group"].isin(classes_per_word[classes_per_word > 1].index)]
    multi_words = {w: g["class"].value_counts().to_dict() for w, g in multi.groupby("group")}
    innate_words = sorted(df.loc[df["class"] == "innate", "group"].unique())
    return {
        "rows_total": len(df),
        "rows_by_class_axis": {c: by_class_axis.loc[c].to_dict() for c in by_class_axis.index},
        "unique_words_by_class": unique.to_dict(),
        "unique_words_by_class_axis": {c: unique_axis.loc[c].to_dict() for c in unique_axis.index},
        "words_in_multiple_classes": multi_words,
        "innate_words": innate_words,
    }, by_class_axis, unique_axis


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    ds = load_items(cfg)
    src_df, report = sentence_rows(ds, cfg)

    if cfg["input_mode"] == "sentence":
        df = src_df
    elif cfg["input_mode"] == "template":
        df, tpl_report = template_rows(src_df, cfg)
        report["template"] = tpl_report
    else:
        raise ValueError(f"unknown input_mode: {cfg['input_mode']}")

    # Spans must point at the word they claim to.
    assert all(s[a:b] == w for s, a, b, w in
               zip(df["sentence"], df["span_start"], df["span_end"], df["word"]))

    summary, by_class_axis, unique_axis = summarize(df, src_df)
    report = {"input_mode": cfg["input_mode"], "dataset_revision": cfg["dataset_revision"],
              **report, **summary}
    df.to_csv(path(cfg, "rows"), index=False)
    write_json(report, path(cfg, "data_report"))

    print(f"input_mode = {cfg['input_mode']}   rows = {len(df)}\n")
    print("Rows per class x axis:\n", by_class_axis, "\n", sep="")
    print("Unique words per class x axis:\n", unique_axis, "\n", sep="")
    print("Unique words per class:", summary["unique_words_by_class"])
    n_innate = summary["unique_words_by_class"].get("innate", 0)
    if not 25 <= n_innate <= 100:
        print(f"  WARNING: {n_innate} unique innate words; handoff expects ~50.")
    print(f"Words labeled with more than one class: {len(summary['words_in_multiple_classes'])}")
    print("\nDrops (reason: axis counts):")
    for reason, counts in report["drops"].items():
        print(f"  {reason}: {counts}  (total {sum(counts.values())})")
    print("\nStats:")
    for k, v in report["stats"].items():
        print(f"  {k}: {v}")
    if "template" in report:
        print(f"\nTemplate mode: words dropped for class ties: "
              f"{len(report['template']['words_dropped_class_tie'])}")
    print(f"\nWrote {path(cfg, 'rows')} and {path(cfg, 'data_report')}")


if __name__ == "__main__":
    main()
