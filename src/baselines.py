"""Stage 4: baselines on the same folds and metrics as train.py.

  single_direction  cosine to the training-fold innate centroid of the row's axis -> tree (depth 2)
  part_of_speech    spaCy POS of the span's head word in its sentence, one-hot -> tree
  frequency         wordfreq Zipf frequency of the word -> tree (depth 2)
  context_only      the chosen three-class forest (train_report.json) on context_embeddings.npy
                    (the sentence without the word)
  logistic_regression  StandardScaler -> LogisticRegression on the span embeddings, C tuned (lr_grid)
  knn                  StandardScaler -> KNeighborsClassifier on the span embeddings, k tuned (knn_grid)

Writes results/baselines_report.json, results/baseline_sweeps.csv and results/comparison.csv (with
both forests from train_report.json). Only dev rows are used; test rows are never read here (see
final.py).
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from wordfreq import zipf_frequency

from utils import (CLASSES, collapsed_f1, dev_folds, direction_feature, evaluate, forest,
                   load_config, one_hot, path, pos_tags, set_seed, write_json)

TIE = 0.001


def single_direction(X, rows, seed):
    y = rows["class"].to_numpy()
    axis = rows["axis"].to_numpy()

    def fit_predict(tr, te):
        # Centroid from the training fold only; feature for this fold's rows only.
        f = direction_feature(X, y, axis, tr, np.concatenate([tr, te]))
        clf = DecisionTreeClassifier(max_depth=2, random_state=seed).fit(f[tr], y[tr])
        return clf.predict(f[tr]), clf.predict(f[te])
    return fit_predict


def tree_on(F, y, seed, **kw):
    def fit_predict(tr, te):
        clf = DecisionTreeClassifier(random_state=seed, **kw).fit(F[tr], y[tr])
        return clf.predict(F[tr]), clf.predict(F[te])
    return fit_predict


def model_on(X, y, make, warned):
    """fit_predict for make(): a fresh estimator per fold, fit on training-fold rows only.

    Appends to `warned`, per fold, whether fitting raised a ConvergenceWarning.
    """
    def fit_predict(tr, te):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = make().fit(X[tr], y[tr])
        warned.append(any(issubclass(w.category, ConvergenceWarning) for w in caught))
        return clf.predict(X[tr]), clf.predict(X[te])
    return fit_predict


def lr(seed, C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, random_state=seed))


def knn(n_neighbors):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=n_neighbors))


def sweep(name, param, values, make, X, y, folds, simpler):
    """CV every grid value; choose the best mean macro-F1, ties within TIE to the simpler model.

    Returns the chosen value, its report (with collapsed F1), and one baseline_sweeps.csv row per value.
    """
    table, reports = [], {}
    for v in values:
        warned = []
        report, oof, _ = evaluate(model_on(X, y, lambda: make(v), warned), y, folds)
        report["collapsed_bias_vs_neutral_f1"] = collapsed_f1(y, oof, folds)
        report["convergence_warning_folds"] = [k for k, w in enumerate(warned) if w]
        reports[v] = report
        s = report["summary"]
        table.append({"model": name, "param": param, "value": v,
                      "macro_f1_mean": s["test_macro_f1"]["mean"],
                      "macro_f1_std": s["test_macro_f1"]["std"],
                      **{f"f1_{c}": s["per_class"][c]["f1-score"]["mean"] for c in CLASSES},
                      "train_macro_f1_mean": s["train_macro_f1"]["mean"],
                      "bias_vs_neutral_f1": report["collapsed_bias_vs_neutral_f1"]["mean"],
                      "convergence_warning_folds": sum(warned)})
        print(f"  {name} {param}={v} done")
    best = max(r["macro_f1_mean"] for r in table)
    chosen = min((r["value"] for r in table if r["macro_f1_mean"] >= best - TIE), key=simpler)
    for r in table:
        r["chosen"] = r["value"] == chosen
    return chosen, reports[chosen], table


def comparison_row(name, report):
    s = report["summary"]
    return {"model": name,
            "macro_f1_mean": s["test_macro_f1"]["mean"], "macro_f1_std": s["test_macro_f1"]["std"],
            **{f"f1_{c}": s["per_class"][c]["f1-score"]["mean"] for c in CLASSES},
            "train_macro_f1_mean": s["train_macro_f1"]["mean"],
            "bias_vs_neutral_f1": report["collapsed_bias_vs_neutral_f1"]["mean"]}


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
    forest_report = json.load(open(out / "train_report.json"))  # run train.py first
    best_rf = forest_report["best_params"]["three_class"]
    dev_idx, test_idx, folds = dev_folds(rows, cfg)
    assert not np.isin(np.concatenate([np.concatenate(f) for f in folds]), test_idx).any(), \
        "a fold index is in the test set"
    print(f"Group assertion: no group in both train and test in any of {len(folds)} dev folds: OK")
    print("Test assertion: no fold index is in the test set: OK")

    # Features for dev rows only; test rows stay zero and are never in a fold.
    tags = pos_tags(rows.iloc[dev_idx])
    vocab = sorted(set(tags))
    pos = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    pos[dev_idx] = one_hot(tags, vocab)
    freq = np.zeros((len(rows), 1))
    freq[dev_idx, 0] = [zipf_frequency(w, "en") for w in rows["word"].iloc[dev_idx]]
    models = {
        "single_direction": single_direction(X, rows, seed),
        "part_of_speech": tree_on(pos, y, seed),
        "frequency": tree_on(freq, y, seed, max_depth=2),
        "context_only": model_on(C, y, lambda: forest(seed, **best_rf), []),
    }

    reports = {}
    for name, fit_predict in models.items():
        reports[name], oof, _ = evaluate(fit_predict, y, folds)
        reports[name]["collapsed_bias_vs_neutral_f1"] = collapsed_f1(y, oof, folds)
        print(f"  {name} done")
    reports["part_of_speech"]["pos_tag_counts"] = pd.Series(tags).value_counts().to_dict()
    reports["context_only"]["params"] = best_rf

    lr_C, reports["logistic_regression"], lr_table = sweep(
        "logistic_regression", "C", cfg["lr_grid"]["C"], lambda C: lr(seed, C), X, y, folds,
        simpler=lambda C: C)
    k, reports["knn"], knn_table = sweep(
        "knn", "n_neighbors", cfg["knn_grid"]["n_neighbors"], knn, X, y, folds,
        simpler=lambda k: -k)
    reports["logistic_regression"]["params"] = {"C": lr_C}
    reports["knn"]["params"] = {"n_neighbors": k}
    write_json(reports, out / "baselines_report.json")
    sweeps = pd.DataFrame(lr_table + knn_table)
    sweeps.to_csv(out / "baseline_sweeps.csv", index=False)

    table = [comparison_row("random_forest", forest_report),
             {"model": "random_forest_binary",
              "bias_vs_neutral_f1": forest_report["binary_rf"]["val_bias_vs_neutral_f1"]["mean"]}]
    table += [comparison_row(name, r) for name, r in reports.items()]
    table = pd.DataFrame(table)
    table.to_csv(out / "comparison.csv", index=False)

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print("\nLR and k-NN grids (CV on the dev set)")
        print(sweeps.to_string(index=False))
        warned = sweeps[(sweeps["model"] == "logistic_regression")
                        & (sweeps["convergence_warning_folds"] > 0)]["value"].tolist()
        print(f"LR convergence warnings at C = {warned or 'none'}")
        print(f"Chosen: logistic_regression C={lr_C}, knn n_neighbors={k};"
              f" context_only forest {best_rf}")
        print("\n", table.to_string(index=False), sep="")
    print(f"\nWrote {out}/baselines_report.json, baseline_sweeps.csv and comparison.csv")


if __name__ == "__main__":
    main()
