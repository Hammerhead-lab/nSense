"""Stage 3: rows + span embeddings -> random forest c(e(o)), 5-fold CV by `group` on the dev set.

For every rf_grid configuration, a three-class RF (trained on `class`) and a binary RF (trained on
to_binary(class)) are cross-validated on the same folds. Each is selected on its own native metric.

Writes to results/: split.csv, split_report.json, rf_sweep.csv, rf_sweep.{png,pdf},
label_gap.{png,pdf}, and, for the chosen three-class configuration, train_report.json,
importances.csv, oof_predictions.csv. Test rows are never read here (see final.py).
"""
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from utils import (CLASSES, bias_score, binary_metrics, collapsed_f1, dev_folds, evaluate,
                   fold_metrics, forest, load_config, path, set_seed, split_report, stat,
                   to_binary, write_json)

matplotlib.use("Agg")

TIE = 0.001
# Categorical slots 1-4 of the dataviz reference palette, in fixed order; markers double-encode.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, GRID = "#0b0b0b", "#e4e3df"
FIG_SIZE = (3.3, 2.2)


def depth_value(d):
    """max_depth as written in rf_sweep.csv."""
    return "none" if d is None else str(d)


def depth_label(d):
    """max_depth as shown in the figures."""
    return "∞" if d is None else str(d)


def sweep_config(X, y, folds, seed, params):
    """5-fold CV of both RFs at one configuration; returns one rf_sweep.csv row."""
    yb = to_binary(y)
    three, binary = [], []
    for tr, te in folds:
        clf = forest(seed, **params).fit(X[tr], y[tr])
        p_te = clf.predict(X[te])
        m = fold_metrics(y[tr], clf.predict(X[tr]), y[te], p_te)
        m["collapsed"] = binary_metrics(y[te], to_binary(p_te))["macro_f1"]
        three.append(m)
        clf = forest(seed, **params).fit(X[tr], yb[tr])
        binary.append({"train": binary_metrics(y[tr], clf.predict(X[tr]))["macro_f1"],
                       "val": binary_metrics(y[te], clf.predict(X[te]))["macro_f1"]})

    cols = {
        "three_val_macro_f1": [f["test_macro_f1"] for f in three],
        "three_train_macro_f1": [f["train_macro_f1"] for f in three],
        **{f"three_val_f1_{c}": [f["per_class"][c]["f1-score"] for f in three] for c in CLASSES},
        "three_collapsed_bvn_f1": [f["collapsed"] for f in three],
        "binary_val_bvn_f1": [f["val"] for f in binary],
        "binary_train_bvn_f1": [f["train"] for f in binary],
        "gap": [t["collapsed"] - b["val"] for t, b in zip(three, binary)],
    }
    row = {"n_estimators": params["n_estimators"], "max_depth": depth_value(params["max_depth"])}
    for name, v in cols.items():
        s = stat(v)
        row[f"{name}_mean"], row[f"{name}_std"] = s["mean"], s["std"]
    return row


def binary_auc_cv(X, y, folds, seed, params):
    """Per-fold bias-vs-neutral AUC and macro-F1 of the binary RF, refit at `params`."""
    yb = to_binary(y)
    auc, f1 = [], []
    for tr, te in folds:
        clf = forest(seed, **params).fit(X[tr], yb[tr])
        auc.append(roc_auc_score(yb[te] == "bias", bias_score(clf.predict_proba(X[te]),
                                                              clf.classes_)))
        f1.append(binary_metrics(y[te], clf.predict(X[te]))["macro_f1"])
    return auc, f1


def select(rows, params, col):
    """Highest mean of `col`; within TIE of the best, fewer trees, then shallower."""
    best = max(r[col] for r in rows)
    tied = [k for k, r in enumerate(rows) if r[col] >= best - TIE]
    k = min(tied, key=lambda k: (params[k]["n_estimators"],
                                 np.inf if params[k]["max_depth"] is None else params[k]["max_depth"]))
    return params[k], rows[k]


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=6.5)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def plot_sweep(sweep, grid, out):
    """RQ2: CV three-class macro-F1 by n_estimators, one line per max_depth; train dashed."""
    trees, depths = grid["n_estimators"], grid["max_depth"]
    fig, ax = plt.subplots(figsize=FIG_SIZE, layout="constrained")
    x = np.arange(len(trees))
    for j, d in enumerate(depths):
        part = sweep[sweep["max_depth"] == depth_value(d)].set_index("n_estimators").loc[trees]
        xs = x + (j - (len(depths) - 1) / 2) * 0.06
        ax.errorbar(xs, part["three_val_macro_f1_mean"], yerr=part["three_val_macro_f1_std"],
                    color=COLORS[j], marker=MARKERS[j], markersize=3.5, linewidth=1.2,
                    capsize=1.5, elinewidth=0.8, label=f"depth {depth_label(d)}")
        ax.plot(xs, part["three_train_macro_f1_mean"], color=COLORS[j], linestyle="--",
                linewidth=1, marker=MARKERS[j], markersize=2.5, markerfacecolor="none")
    ax.plot([], [], color=INK, linewidth=1.2, label="validation")
    ax.plot([], [], color=INK, linewidth=1, linestyle="--", label="train")
    ax.set_xticks(x, [str(t) for t in trees])
    ax.set_xlabel("n_estimators", fontsize=7, color=INK)
    ax.set_ylabel("3-class macro-F1 (CV)", fontsize=7, color=INK)
    style(ax)
    fig.legend(loc="outside right center", fontsize=6, frameon=False, handlelength=1.6)
    save(fig, out, "rf_sweep")


def plot_gap(sweep, grid, out):
    """RQ1 across settings: CV bias-vs-neutral F1, binary RF vs collapsed three-class RF."""
    order = [(d, t) for d in grid["max_depth"] for t in grid["n_estimators"]]
    part = sweep.set_index(["max_depth", "n_estimators"]) \
        .loc[[(depth_value(d), t) for d, t in order]]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=FIG_SIZE, layout="constrained")
    for j, (col, label) in enumerate([("binary_val_bvn_f1", "binary RF"),
                                      ("three_collapsed_bvn_f1", "3-class RF, collapsed")]):
        ax.errorbar(x + (j - 0.5) * 0.15, part[f"{col}_mean"], yerr=part[f"{col}_std"],
                    color=COLORS[j], marker=MARKERS[j], markersize=3, linewidth=1.2,
                    capsize=1.5, elinewidth=0.8, label=label)
    ax.set_xticks(x, [f"d{depth_label(d)}·{t}" for d, t in order], rotation=90)
    ax.set_xlabel("max_depth · n_estimators", fontsize=7, color=INK)
    ax.set_ylabel("Bias-vs-neutral F1 (CV)", fontsize=7, color=INK)
    style(ax)
    fig.legend(loc="outside upper center", ncol=2, fontsize=6, frameon=False)
    save(fig, out, "label_gap")


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
    dev_idx, test_idx, folds = dev_folds(rows, cfg)
    assert not np.isin(np.concatenate([np.concatenate(f) for f in folds]), test_idx).any(), \
        "a fold index is in the test set"
    split_report(rows, cfg, dev_idx, test_idx, out)
    print(f"Dev/test split: {len(dev_idx)} dev rows, {len(test_idx)} test rows; no shared group: OK")
    print(f"Group assertion: no group in both train and test in any of {len(folds)} dev folds: OK")
    print("Test assertion: no fold index is in the test set: OK")

    grid = cfg["rf_grid"]
    params = [{"n_estimators": t, "max_depth": d}
              for d, t in itertools.product(grid["max_depth"], grid["n_estimators"])]
    sweep = []
    for p in params:
        sweep.append(sweep_config(X, y, folds, seed, p))
        print(f"  sweep d{depth_label(p['max_depth'])}·{p['n_estimators']} done")
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out / "rf_sweep.csv", index=False)
    plot_sweep(sweep_df, grid, out)
    plot_gap(sweep_df, grid, out)

    best3, row3 = select(sweep, params, "three_val_macro_f1_mean")
    best2, row2 = select(sweep, params, "binary_val_bvn_f1_mean")

    importances, auc3 = [], []

    def fit_predict(tr, te):
        clf = forest(seed, **best3).fit(X[tr], y[tr])
        imp = permutation_importance(clf, X[te], y[te], scoring="f1_macro", n_repeats=5,
                                     random_state=seed, n_jobs=-1)
        importances.append(imp.importances_mean)
        auc3.append(roc_auc_score(to_binary(y[te]) == "bias",
                                  bias_score(clf.predict_proba(X[te]), clf.classes_)))
        print(f"  fold {len(importances) - 1} done")
        return clf.predict(X[tr]), clf.predict(X[te])

    report, oof, fold_of = evaluate(fit_predict, y, folds)
    collapsed = collapsed_f1(y, oof, folds)
    assert abs(collapsed["mean"] - row3["three_collapsed_bvn_f1_mean"]) < 1e-12, \
        "chosen-configuration refit does not match the sweep"
    auc2, f1_2 = binary_auc_cv(X, y, folds, seed, best2)
    assert abs(stat(f1_2)["mean"] - row2["binary_val_bvn_f1_mean"]) < 1e-12 and \
        abs(stat(f1_2)["std"] - row2["binary_val_bvn_f1_std"]) < 1e-12, \
        "binary refit does not match the sweep"
    diff = np.array(auc3) - np.array(auc2)
    rq1_auc_cv = {"three_class_collapsed": stat(auc3), "binary": stat(auc2),
                  "paired_difference": stat(diff),
                  "folds": [{"three_class_collapsed": a3, "binary": a2, "difference": float(d)}
                            for a3, a2, d in zip(auc3, auc2, diff)]}
    write_json({"model": "random_forest", "n_rows": len(dev_idx),
                "best_params": {"three_class": best3, "binary": best2},
                "selection": {"three_class": "max mean CV 3-class macro-F1",
                              "binary": "max mean CV bias-vs-neutral macro-F1",
                              "tie_break": f"within {TIE}: fewer trees, then shallower"},
                "collapsed_bias_vs_neutral_f1": collapsed,
                "binary_rf": {"val_bias_vs_neutral_f1": {"mean": row2["binary_val_bvn_f1_mean"],
                                                         "std": row2["binary_val_bvn_f1_std"]},
                              "train_bias_vs_neutral_f1": {"mean": row2["binary_train_bvn_f1_mean"],
                                                           "std": row2["binary_train_bvn_f1_std"]}},
                "rq1_auc_cv": rq1_auc_cv,
                **report},
               out / "train_report.json")

    imp = np.array(importances)
    pd.DataFrame({"feature": np.arange(X.shape[1]), "importance_mean": imp.mean(axis=0),
                  "importance_std_over_folds": imp.std(axis=0)}) \
        .sort_values("importance_mean", ascending=False) \
        .to_csv(out / "importances.csv", index=False)
    pd.DataFrame({"row": np.arange(len(rows)), "item_id": rows["item_id"], "word": rows["word"],
                  "group": rows["group"], "true": y, "pred": oof, "fold": fold_of}) \
        .iloc[dev_idx].to_csv(out / "oof_predictions.csv", index=False)

    show = ["max_depth", "n_estimators", "three_val_macro_f1_mean", "three_val_macro_f1_std",
            "three_train_macro_f1_mean", "three_collapsed_bvn_f1_mean", "binary_val_bvn_f1_mean",
            "binary_train_bvn_f1_mean", "gap_mean", "gap_std"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print("\nRF sweep (CV on the dev set; bvn = bias vs neutral)")
        print(sweep_df[show].to_string(index=False))
    print(f"\nChosen 3-class RF: {best3}   chosen binary RF: {best2}")

    s = report["summary"]
    print(f"\nThree-class random forest {best3}, {len(folds)}-fold grouped CV on the dev set")
    print(f"  {'fold':>4}  {'train F1':>8}  {'held-out F1':>11}")
    for k, f in enumerate(report["folds"]):
        print(f"  {k:>4}  {f['train_macro_f1']:8.3f}  {f['test_macro_f1']:11.3f}")
    print(f"  mean  {s['train_macro_f1']['mean']:8.3f}  {s['test_macro_f1']['mean']:11.3f}"
          f"  (held-out std {s['test_macro_f1']['std']:.3f})")
    print("  per-class held-out F1:",
          {c: round(s["per_class"][c]["f1-score"]["mean"], 3) for c in CLASSES})
    print(f"  collapsed bias-vs-neutral F1: {collapsed['mean']:.3f} ± {collapsed['std']:.3f}")
    print(f"  confusion matrix (rows true, cols pred, {CLASSES}):")
    for c, r in zip(CLASSES, s["confusion_matrix_total"]):
        print(f"    {c:>8} {r}")

    print("\nRQ1 bias-vs-neutral AUC, CV on the dev set (chosen configurations)")
    print(f"  {'fold':>4}  {'3-class':>8}  {'binary':>8}  {'3-class - binary':>16}")
    for k, f in enumerate(rq1_auc_cv["folds"]):
        print(f"  {k:>4}  {f['three_class_collapsed']:8.4f}  {f['binary']:8.4f}"
              f"  {f['difference']:+16.4f}")
    a3, a2, d = (rq1_auc_cv[k] for k in ("three_class_collapsed", "binary", "paired_difference"))
    print(f"  mean  {a3['mean']:8.4f}  {a2['mean']:8.4f}  {d['mean']:+16.4f}")
    print(f"  std   {a3['std']:8.4f}  {a2['std']:8.4f}  {d['std']:16.4f}")
    print(f"\nWrote {out}/split.csv, split_report.json, rf_sweep.csv, rf_sweep.png/.pdf,"
          " label_gap.png/.pdf, train_report.json, importances.csv, oof_predictions.csv")


if __name__ == "__main__":
    main()
