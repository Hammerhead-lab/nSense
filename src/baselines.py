"""Stage 4: baselines on the same folds and metrics as train.py.

  single_direction  cosine to the training-fold innate centroid of the row's axis -> tree (depth 2)
  part_of_speech    spaCy POS of the span's head word in its sentence, one-hot -> tree
  frequency         wordfreq Zipf frequency of the word -> tree (depth 2)
  context_only      the forest's settings on context_embeddings.npy (the sentence without the word)

Writes results/baselines_report.json and results/comparison.csv (with the forest from train_report.json).
"""
import json

import numpy as np
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from wordfreq import zipf_frequency

from utils import CLASSES, evaluate, group_folds, load_config, path, set_seed, write_json

SPACY_MODEL = "en_core_web_sm"


def single_direction(X, rows, seed):
    y = rows["class"].to_numpy()
    axis = rows["axis"].to_numpy()
    innate = y == "innate"
    unit = X / np.linalg.norm(X, axis=1, keepdims=True)

    def fit_predict(tr, te):
        feat = np.zeros(len(X))
        for a in np.unique(axis):
            c = X[tr][innate[tr] & (axis[tr] == a)].mean(axis=0)  # training fold only
            sel = axis == a
            feat[sel] = unit[sel] @ (c / np.linalg.norm(c))
        f = feat[:, None]
        clf = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(f[tr], y[tr])
        return clf.predict(f[tr]), clf.predict(f[te])
    return fit_predict


def pos_features(rows):
    nlp = spacy.load(SPACY_MODEL)
    tags = []
    for doc, r in zip(nlp.pipe(rows["sentence"]), rows.itertuples()):
        span = doc.char_span(r.span_start, r.span_end, alignment_mode="expand")
        tags.append(span.root.pos_)
    return pd.get_dummies(pd.Series(tags, name="pos")).to_numpy(dtype=np.float32), tags


def tree_on(F, y, seed, **kw):
    def fit_predict(tr, te):
        clf = DecisionTreeClassifier(random_state=seed, **kw).fit(F[tr], y[tr])
        return clf.predict(F[tr]), clf.predict(F[te])
    return fit_predict


def forest_on(X, y, seed):
    def fit_predict(tr, te):
        clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        return clf.predict(X[tr]), clf.predict(X[te])
    return fit_predict


def comparison_row(name, report):
    s = report["summary"]
    return {"model": name,
            "macro_f1_mean": s["test_macro_f1"]["mean"], "macro_f1_std": s["test_macro_f1"]["std"],
            **{f"f1_{c}": s["per_class"][c]["f1-score"]["mean"] for c in CLASSES},
            "train_macro_f1_mean": s["train_macro_f1"]["mean"]}


def main():
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    out = path(cfg, "results")
    out.mkdir(exist_ok=True)

    rows = pd.read_csv(path(cfg, "rows"), keep_default_na=False)
    X = np.load(path(cfg, "embeddings"))
    C = np.load(path(cfg, "context_embeddings"))
    assert len(X) == len(C) == len(rows), "embeddings do not match rows.csv; rerun embed.py"
    y = rows["class"].to_numpy()
    folds = group_folds(rows["group"], cfg["n_splits"])
    print(f"Group assertion: no group in both train and test in any of {len(folds)} folds: OK")

    pos, tags = pos_features(rows)
    freq = np.array([[zipf_frequency(w, "en")] for w in rows["word"]])
    models = {
        "single_direction": single_direction(X, rows, seed),
        "part_of_speech": tree_on(pos, y, seed),
        "frequency": tree_on(freq, y, seed, max_depth=2),
        "context_only": forest_on(C, y, seed),
    }

    reports = {}
    for name, fit_predict in models.items():
        reports[name], _, _ = evaluate(fit_predict, y, folds)
        print(f"  {name} done")
    reports["part_of_speech"]["pos_tag_counts"] = pd.Series(tags).value_counts().to_dict()
    write_json(reports, out / "baselines_report.json")

    table = []
    forest_report = out / "train_report.json"
    if forest_report.exists():
        table.append(comparison_row("random_forest", json.load(open(forest_report))))
    else:
        print("  NOTE: results/train_report.json missing; run train.py for the forest row")
    table += [comparison_row(name, r) for name, r in reports.items()]
    table = pd.DataFrame(table)
    table.to_csv(out / "comparison.csv", index=False)

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print("\n", table.to_string(index=False), sep="")
    print(f"\nWrote {out}/baselines_report.json and comparison.csv")


if __name__ == "__main__":
    main()
