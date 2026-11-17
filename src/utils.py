"""Shared helpers: config loading, paths, seeding, and the folds and metrics every model uses."""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["innate", "derived", "neutral"]
BINARY = ["bias", "neutral"]
SPACY_MODEL = "en_core_web_sm"


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def path(cfg, key):
    """Absolute path for an entry under `paths:` in the config."""
    p = ROOT / cfg["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def group_folds(groups, n_splits):
    """Balance groups by size; break group ties alphabetically and fold ties by index."""
    groups = np.asarray(groups)
    counts = Counter(groups)
    if not 2 <= n_splits <= len(counts):
        raise ValueError("n_splits must be between 2 and the number of groups")
    sizes = [0] * n_splits
    mapping = {}
    for group in sorted(counts, key=lambda g: (-counts[g], g)):
        fold = min(range(n_splits), key=lambda k: (sizes[k], k))
        mapping[group] = fold
        sizes[fold] += counts[group]
    fold_ids = np.array([mapping[g] for g in groups])
    folds = [(np.flatnonzero(fold_ids != k), np.flatnonzero(fold_ids == k))
             for k in range(n_splits)]
    for k, (tr, te) in enumerate(folds):
        overlap = set(groups[tr]) & set(groups[te])
        assert not overlap, f"fold {k}: groups in both train and test: {sorted(overlap)[:10]}"
    return folds


def test_fold_candidates(rows, cfg):
    """Every held-out fold of a StratifiedGroupKFold on class, by group, scored on labels only.

    max_gap is the largest |class share in the fold - overall class share| over the classes, in
    percentage points. A fold is eligible if its test share is within 15-25%.
    """
    y = rows["class"].to_numpy()
    overall = {c: 100 * np.mean(y == c) for c in CLASSES}
    sgkf = StratifiedGroupKFold(n_splits=cfg["test_n_splits"], shuffle=True,
                                random_state=cfg["seed"])
    candidates = []
    for k, (dev_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(rows)), y, rows["group"])):
        share = {c: 100 * np.mean(y[test_idx] == c) for c in CLASSES}
        test_share = len(test_idx) / len(rows)
        candidates.append({"fold": k, "test_share": test_share, "class_pct": share,
                           "max_gap": max(abs(share[c] - overall[c]) for c in CLASSES),
                           "eligible": 0.15 <= test_share <= 0.25,
                           "dev_idx": dev_idx, "test_idx": test_idx})
    return overall, candidates


def dev_test_split(rows, cfg):
    """Lock the test set: the eligible candidate fold with the smallest max_gap (ties: lowest index)."""
    groups = rows["group"].to_numpy()
    _, candidates = test_fold_candidates(rows, cfg)
    eligible = [c for c in candidates if c["eligible"]]
    assert eligible, "no candidate test fold has a test share within 15-25%"
    best = min(eligible, key=lambda c: (c["max_gap"], c["fold"]))
    dev_idx, test_idx = best["dev_idx"], best["test_idx"]
    overlap = set(groups[dev_idx]) & set(groups[test_idx])
    assert not overlap, f"groups in both dev and test: {sorted(overlap)[:10]}"
    return dev_idx, test_idx


def dev_folds(rows, cfg):
    """Test split, then group_folds on the dev rows only, with fold indices mapped back to rows."""
    dev_idx, test_idx = dev_test_split(rows, cfg)
    folds = [(dev_idx[tr], dev_idx[te])
             for tr, te in group_folds(rows["group"].to_numpy()[dev_idx], cfg["n_splits"])]
    return dev_idx, test_idx, folds


def split_report(rows, cfg, dev_idx, test_idx, out):
    """Write results/split.csv and results/split_report.json; return the report."""
    overall, candidates = test_fold_candidates(rows, cfg)
    chosen = [c["fold"] for c in candidates if np.array_equal(c["test_idx"], test_idx)]
    assert len(chosen) == 1, "test_idx is not one of the candidate folds"
    split = np.full(len(rows), "", dtype=object)
    split[dev_idx], split[test_idx] = "dev", "test"
    df = rows[["item_id", "word", "group", "axis", "class"]].assign(split=split)
    df.insert(0, "row", np.arange(len(rows)))
    df.to_csv(out / "split.csv", index=False)

    report = {"test_share": len(test_idx) / len(rows), "chosen_test_fold": chosen[0],
              "overall_class_pct": overall,
              "candidate_test_folds": [{k: c[k] for k in ("fold", "test_share", "class_pct",
                                                          "max_gap", "eligible")}
                                       for c in candidates],
              "splits": {}}
    for name in ("dev", "test"):
        part = df[df["split"] == name]
        counts = part["class"].value_counts()
        report["splits"][name] = {
            "rows": len(part),
            "groups": int(part["group"].nunique()),
            "class_counts": {c: int(counts.get(c, 0)) for c in CLASSES},
            "class_pct": {c: 100 * counts.get(c, 0) / len(part) for c in CLASSES},
            "class_by_axis": {c: {a: int(n) for a, n in
                                  part[part["class"] == c]["axis"].value_counts().sort_index().items()}
                              for c in CLASSES},
        }
    totals = df["class"].value_counts()
    report["class_test_share"] = {c: report["splits"]["test"]["class_counts"][c] / int(totals[c])
                                  for c in CLASSES}
    write_json(report, out / "split_report.json")
    return report


def to_binary(labels):
    """Map three-class labels to {bias, neutral} with `binary_map` from config.yaml."""
    m = load_config()["binary_map"]
    return np.array([m[l] for l in labels], dtype=object)


def binary_metrics(y_true3, y_pred_binary):
    """Bias-vs-neutral metrics: the true three-class labels are mapped with to_binary first."""
    y_true = to_binary(y_true3)
    rep = classification_report(y_true, y_pred_binary, labels=BINARY, output_dict=True,
                                zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred_binary, labels=BINARY, average="macro", zero_division=0),
        "per_class": {c: {m: rep[c][m] for m in ("precision", "recall", "f1-score", "support")}
                      for c in BINARY},
        "confusion_matrix": confusion_matrix(y_true, y_pred_binary, labels=BINARY).tolist(),
    }


def bias_score(proba, classes):
    """P(bias) from predict_proba: the sum of the columns whose class maps to `bias`.

    Three-class model: P(innate) + P(derived). Binary model: P(bias). Columns are found by name in
    `classes` (the model's classes_), never by position.
    """
    m = load_config()["binary_map"]
    classes = list(classes)
    assert proba.shape[1] == len(classes), "predict_proba columns do not match classes_"
    assert all(c in m or c in BINARY for c in classes), f"unknown classes {classes}"
    cols = [k for k, c in enumerate(classes) if m.get(c, c) == "bias"]
    assert cols, f"no bias class in {classes}"
    return proba[:, cols].sum(axis=1)


def stat(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std())}


def collapsed_f1(y, oof, folds):
    """Mean and std over folds of the bias-vs-neutral macro-F1 of three-class OOF predictions."""
    y = np.asarray(y)
    return stat([binary_metrics(y[te], to_binary(oof[te]))["macro_f1"] for _, te in folds])


def forest(seed, n_estimators, max_depth):
    """The random forest c, at one rf_grid configuration; everything else is the default."""
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  random_state=seed, n_jobs=-1)


def direction_feature(X, y, axis, fit_idx, idx):
    """Cosine of rows `idx` to the innate centroid of their own axis, from rows `fit_idx` only.

    Returns a (len(X), 1) feature; rows outside `idx` stay zero.
    """
    feat = np.zeros(len(X))
    for a in np.unique(axis):
        c = X[fit_idx][(y[fit_idx] == "innate") & (axis[fit_idx] == a)].mean(axis=0)
        sel = idx[axis[idx] == a]
        unit = X[sel] / np.linalg.norm(X[sel], axis=1, keepdims=True)
        feat[sel] = unit @ (c / np.linalg.norm(c))
    return feat[:, None]


def pos_tags(rows):
    """spaCy POS of each span's head word, tagged in its full sentence."""
    import spacy
    nlp = spacy.load(SPACY_MODEL)
    tags = []
    for doc, r in zip(nlp.pipe(rows["sentence"]), rows.itertuples()):
        span = doc.char_span(r.span_start, r.span_end, alignment_mode="expand")
        tags.append(span.root.pos_)
    return tags


def one_hot(tags, vocab):
    """One-hot over `vocab` (the tags seen in the fitting rows); unseen tags are all zero."""
    return np.array([[t == v for v in vocab] for t in tags], dtype=np.float32)


def fold_metrics(y_tr, p_tr, y_te, p_te):
    rep = classification_report(y_te, p_te, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "train_macro_f1": f1_score(y_tr, p_tr, labels=CLASSES, average="macro", zero_division=0),
        "test_macro_f1": f1_score(y_te, p_te, labels=CLASSES, average="macro", zero_division=0),
        "per_class": {c: {m: rep[c][m] for m in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSES},
        "confusion_matrix": confusion_matrix(y_te, p_te, labels=CLASSES).tolist(),
    }


def evaluate(fit_predict, y, folds):
    """Run fit_predict(train_idx, test_idx) -> (train_pred, test_pred) on every fold.

    Returns the report (per fold, mean and std), out-of-fold predictions and each row's fold.
    """
    y = np.asarray(y)
    per_fold, oof, fold_of = [], np.empty(len(y), dtype=object), np.full(len(y), -1)
    for k, (tr, te) in enumerate(folds):
        p_tr, p_te = fit_predict(tr, te)
        per_fold.append(fold_metrics(y[tr], p_tr, y[te], p_te))
        oof[te], fold_of[te] = p_te, k

    def over_folds(get):
        return stat([get(f) for f in per_fold])

    summary = {"train_macro_f1": over_folds(lambda f: f["train_macro_f1"]),
               "test_macro_f1": over_folds(lambda f: f["test_macro_f1"]),
               "per_class": {c: {m: over_folds(lambda f: f["per_class"][c][m])
                                 for m in ("precision", "recall", "f1-score")} for c in CLASSES},
               "confusion_matrix_total": np.sum([f["confusion_matrix"] for f in per_fold],
                                                axis=0).tolist()}
    return {"classes": CLASSES, "folds": per_fold, "summary": summary}, oof, fold_of
