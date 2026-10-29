"""Shared helpers: config loading, paths, seeding, and the folds and metrics every model uses."""
import json
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["innate", "derived", "neutral"]


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
    """GroupKFold on `group`. Hard check: no group is in both train and test of any fold."""
    groups = np.asarray(groups)
    folds = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
    for k, (tr, te) in enumerate(folds):
        overlap = set(groups[tr]) & set(groups[te])
        assert not overlap, f"fold {k}: groups in both train and test: {sorted(overlap)[:10]}"
    return folds


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

    def stat(get):
        v = np.array([get(f) for f in per_fold])
        return {"mean": float(v.mean()), "std": float(v.std())}

    summary = {"train_macro_f1": stat(lambda f: f["train_macro_f1"]),
               "test_macro_f1": stat(lambda f: f["test_macro_f1"]),
               "per_class": {c: {m: stat(lambda f: f["per_class"][c][m])
                                 for m in ("precision", "recall", "f1-score")} for c in CLASSES},
               "confusion_matrix_total": np.sum([f["confusion_matrix"] for f in per_fold],
                                                axis=0).tolist()}
    return {"classes": CLASSES, "folds": per_fold, "summary": summary}, oof, fold_of
