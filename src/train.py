"""Stage 3: rows + span embeddings -> random forest c(e(o)), evaluated with GroupKFold on `group`.

Writes to results/: train_report.json, importances.csv, oof_predictions.csv, forest.joblib.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from utils import CLASSES, evaluate, group_folds, load_config, path, set_seed, write_json


def forest(seed):
    return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)


def main():
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    out = path(cfg, "results")
    out.mkdir(exist_ok=True)

    rows = pd.read_csv(path(cfg, "rows"), keep_default_na=False)
    X = np.load(path(cfg, "embeddings"))
    assert len(X) == len(rows), f"{len(X)} embeddings for {len(rows)} rows; rerun embed.py"
    y = rows["class"].to_numpy()
    folds = group_folds(rows["group"], cfg["n_splits"])
    print(f"Group assertion: no group in both train and test in any of {len(folds)} folds: OK")

    importances = []

    def fit_predict(tr, te):
        clf = forest(seed).fit(X[tr], y[tr])
        imp = permutation_importance(clf, X[te], y[te], scoring="f1_macro", n_repeats=5,
                                     random_state=seed, n_jobs=-1)
        importances.append(imp.importances_mean)
        print(f"  fold {len(importances) - 1} done")
        return clf.predict(X[tr]), clf.predict(X[te])

    report, oof, fold_of = evaluate(fit_predict, y, folds)
    write_json({"model": "random_forest", "n_rows": len(rows), **report}, out / "train_report.json")

    imp = np.array(importances)
    pd.DataFrame({"feature": np.arange(X.shape[1]), "importance_mean": imp.mean(axis=0),
                  "importance_std_over_folds": imp.std(axis=0)}) \
        .sort_values("importance_mean", ascending=False) \
        .to_csv(out / "importances.csv", index=False)
    pd.DataFrame({"row": np.arange(len(rows)), "item_id": rows["item_id"], "word": rows["word"],
                  "group": rows["group"], "true": y, "pred": oof, "fold": fold_of}) \
        .to_csv(out / "oof_predictions.csv", index=False)
    joblib.dump(forest(seed).fit(X, y), out / "forest.joblib")

    s = report["summary"]
    print(f"\nRandom forest, {len(folds)}-fold GroupKFold")
    print(f"  {'fold':>4}  {'train F1':>8}  {'held-out F1':>11}")
    for k, f in enumerate(report["folds"]):
        print(f"  {k:>4}  {f['train_macro_f1']:8.3f}  {f['test_macro_f1']:11.3f}")
    print(f"  mean  {s['train_macro_f1']['mean']:8.3f}  {s['test_macro_f1']['mean']:11.3f}"
          f"  (held-out std {s['test_macro_f1']['std']:.3f})")
    print("  per-class held-out F1:",
          {c: round(s["per_class"][c]["f1-score"]["mean"], 3) for c in CLASSES})
    print(f"  confusion matrix (rows true, cols pred, {CLASSES}):")
    for c, r in zip(CLASSES, s["confusion_matrix_total"]):
        print(f"    {c:>8} {r}")
    print(f"\nWrote {out}/train_report.json, importances.csv, oof_predictions.csv, forest.joblib")


if __name__ == "__main__":
    main()
