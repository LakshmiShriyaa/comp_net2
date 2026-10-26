"""Per-OD nonlinear gradient-boosted forecast benchmark.

Uses one regularized histogram gradient-boosting regressor per OD pair. Fit is
restricted to training windows, validation selects among a small configuration
grid, and the held-out test set is scored only for the selected configuration.
"""
import argparse
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from data import load_tm, despike, prepare, metrics, norm_metrics


def features_for(X, tod, scale):
    n = X.shape[-1]
    # Flow-local recent history plus predictable target-time cycles.
    h = (X / scale).transpose(0, 2, 3, 1)
    t = np.broadcast_to(tod[:, None, None, :], (len(X), n, n, tod.shape[1]))
    return np.concatenate([h, t], axis=-1).transpose(1, 2, 0, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--step-min", type=int, default=15)
    ap.add_argument("--season", type=int, default=96)
    ap.add_argument("--scale", choices=("mean", "range", "std"), default="range")
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Implemented below after constructing time-indexed features.
    tm, _ = despike(load_tm(args.data, args.nodes))
    times = np.load(args.times)
    d = prepare(tm, args.window, args.horizon, mask_diag=True, times=times,
                step_min=args.step_min, season=args.season)
    train_tm = d["tm"][:d["t_end"]]
    if args.scale == "mean":
        scale = np.maximum(train_tm.mean(axis=0), 1.0)
    elif args.scale == "range":
        scale = np.maximum(train_tm.max(axis=0) - train_tm.min(axis=0), 1.0)
    else:
        scale = np.maximum(train_tm.std(axis=0), 1.0)
    scale = scale.astype(np.float64)
    n = args.nodes
    bounds = {"train": 0, "val": d["t_end"], "test": d["v_end"]}
    arrays = {}
    for name in ("train", "val", "test"):
        X = d["scaler"].inverse(d[name][0])
        Y = d["scaler"].inverse(d[name][1][:, -1])
        idx = bounds[name] + d[name + "_s"] + args.window + args.horizon - 1
        minute = times[idx]
        dayphase = 2 * np.pi * (minute % 1440) / 1440
        weekphase = 2 * np.pi * ((minute // 1440) % 7) / 7
        tod = np.stack([np.sin(dayphase), np.cos(dayphase),
                        np.sin(weekphase), np.cos(weekphase)], axis=1)
        features = features_for(X, tod, scale).astype(np.float32)
        arrays[name] = (X, Y, features)

    Xtr, Ytr, Ftr = arrays["train"]
    Xv, Yv, Fv = arrays["val"]
    Xt, Yt, Fe = arrays["test"]
    if args.train_stride > 1:
        Xtr, Ytr, Ftr = Xtr[::args.train_stride], Ytr[::args.train_stride], Ftr[:, :, ::args.train_stride]
    mask = ~np.eye(n, dtype=bool)
    lo, hi = train_tm.min(axis=0), train_tm.max(axis=0)

    configs = [
        {"max_iter": 40, "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2_regularization": 1.0},
        {"max_iter": 80, "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2_regularization": 1.0},
        {"max_iter": 80, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 1.0},
    ]

    # Candidate validation models are trained on training data only.
    val_candidates = []
    for config in configs:
        pv = np.zeros((len(Xv), n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = scale[i, j]
                model = HistGradientBoostingRegressor(
                    learning_rate=0.08, early_stopping=False, random_state=0,
                    **config)
                model.fit(Ftr[i, j], (Ytr[:, i, j] - Xtr[:, -1, i, j]) / s)
                pv[:, i, j] = np.maximum(Xv[:, -1, i, j] + model.predict(Fv[i, j]) * s, 0)
        score = norm_metrics(pv, Yv, lo, hi, mask)["RMSE"]
        print(f"validation config={config} normalized_rmse={score:.6f}")
        val_candidates.append((score, config, pv))
    best_score, best_config, pred_val = min(val_candidates, key=lambda a: a[0])

    # Refit chosen configuration on train+validation and evaluate test once.
    Xfit = np.concatenate([Xtr, Xv]); Yfit = np.concatenate([Ytr, Yv])
    Ffit = np.concatenate([Ftr, Fv], axis=2)
    pred_test = np.zeros((len(Xt), n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = scale[i, j]
            model = HistGradientBoostingRegressor(
                learning_rate=0.08, early_stopping=False, random_state=0,
                **best_config)
            # For every window, residuals must be relative to its own last input.
            yfit_delta = (Yfit[:, i, j] - Xfit[:, -1, i, j]) / s
            model.fit(Ffit[i, j], yfit_delta)
            pred_test[:, i, j] = np.maximum(Xt[:, -1, i, j] + model.predict(Fe[i, j]) * s, 0)
    for P in (pred_val, pred_test):
        P[:, np.arange(n), np.arange(n)] = 0

    def score(P, Y):
        return {"original_units": metrics(P, Y, mask),
                "per_pair_normalized": norm_metrics(P, Y, lo, hi, mask)}
    last_val, last_test = Xv[:, -1].copy(), Xt[:, -1].copy()
    for P in (last_val, last_test):
        P[:, np.arange(n), np.arange(n)] = 0
    report = {"flags": vars(args), "selected_on_validation": {"config": best_config,
              "normalized_rmse": best_score},
              "validation": {"naive_last_value": score(last_val, Yv), "flow_tree": score(pred_val, Yv)},
              "test": {"naive_last_value": score(last_test, Yt), "flow_tree": score(pred_test, Yt)}}
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/pred_val.npy", pred_val); np.save(f"{args.out}/true_val.npy", Yv)
    np.save(f"{args.out}/last_observed_val.npy", last_val)
    np.save(f"{args.out}/pred_test.npy", pred_test); np.save(f"{args.out}/true_test.npy", Yt)
    np.save(f"{args.out}/last_observed_test.npy", last_test)
    with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
