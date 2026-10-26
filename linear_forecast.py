"""Validation-selected per-OD autoregressive benchmark for WAN traffic matrices.

Fits a small ridge regression to each OD pair's recent history and predicts the
change from the most recent observed value. Hyperparameters are selected on the
chronological validation split; the test split is evaluated once afterward.

Example (one-hour-ahead forecast on GEANT):
    python linear_forecast.py --data geant_final.npy --nodes 23 --window 12 \
        --horizon 4 --times geant_times.npy --mask-diag --out res_AR
"""
import argparse
import json
import os

import numpy as np

from data import load_tm, despike, prepare, metrics, norm_metrics


def _fit_predict(X_train, Y_train, X_eval, scale, n, features, alpha):
    """Fit one centered ridge model per OD pair using only training windows."""
    # (flows, windows, lag features); normalization makes a single alpha usable
    # across OD pairs whose traffic levels differ by several orders of magnitude.
    xt = (X_train / scale).reshape(len(X_train), X_train.shape[1], -1).transpose(2, 0, 1)
    xe = (X_eval / scale).reshape(len(X_eval), X_eval.shape[1], -1).transpose(2, 0, 1)
    yt = ((Y_train - X_train[:, -1]) / scale).reshape(len(X_train), -1).T
    xt, xe = xt[:, :, -features:], xe[:, :, -features:]

    x_mean = xt.mean(axis=1, keepdims=True)
    y_mean = yt.mean(axis=1, keepdims=True)
    xc = xt - x_mean
    yc = yt - y_mean
    gram = np.einsum("fsi,fsj->fij", xc, xc)
    rhs = np.einsum("fsi,fs->fi", xc, yc)
    eye = np.eye(features, dtype=np.float64)[None, :, :]
    coef = np.linalg.solve(gram + alpha * eye, rhs[..., None])[..., 0]
    intercept = y_mean[:, 0] - np.einsum("fi,fi->f", x_mean[:, 0], coef)
    pred_delta = np.einsum("fsi,fi->fs", xe, coef) + intercept[:, None]
    pred = pred_delta.T * scale.reshape(-1) + X_eval[:, -1].reshape(len(X_eval), -1)
    pred = np.maximum(pred.reshape(-1, n, n), 0.0)
    pred[:, np.arange(n), np.arange(n)] = 0.0
    return pred, coef, intercept


def _fit_predict_selected(X_train, Y_train, X_eval, scale, features_by_flow,
                         alpha_by_flow):
    """Refit validation-selected lag/penalty settings independently per OD pair."""
    n = X_train.shape[-1]
    pred = np.zeros((len(X_eval), n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            k = int(features_by_flow[i, j])
            alpha = float(alpha_by_flow[i, j])
            s = float(scale[i, j])
            x = X_train[:, -k:, i, j] / s
            y = (Y_train[:, i, j] - X_train[:, -1, i, j]) / s
            xm, ym = x.mean(axis=0), y.mean()
            xc, yc = x - xm, y - ym
            coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(k), xc.T @ yc)
            intercept = ym - xm @ coef
            xe = X_eval[:, -k:, i, j] / s
            pred[:, i, j] = np.maximum((xe @ coef + intercept) * s + X_eval[:, -1, i, j], 0.0)
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--times", default=None)
    ap.add_argument("--step-min", type=int, default=15)
    ap.add_argument("--season", type=int, default=96)
    ap.add_argument("--mask-diag", action="store_true")
    ap.add_argument("--no-despike", action="store_true")
    ap.add_argument("--scaling", choices=("mean", "range", "std"), default="mean",
                    help="training-only per-flow scale used by the ridge model")
    ap.add_argument("--selection", choices=("raw_rmse", "raw_mae", "normalized_rmse", "normalized_mae"),
                    default="raw_rmse", help="validation metric used to choose lag count and ridge penalty")
    ap.add_argument("--per-flow-selection", action="store_true",
                    help="choose lag count and penalty separately for each OD flow using validation normalized RMSE")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tm = load_tm(args.data, args.nodes)
    if not args.no_despike:
        tm, fixed = despike(tm)
        print(f"Repaired {len(fixed)} gross outlier step(s): {fixed[:20]}")
    times = np.load(args.times) if args.times else None
    d = prepare(tm, args.window, args.horizon, mask_diag=args.mask_diag,
                times=times, step_min=args.step_min, season=args.season)
    print("Windows:", d["stats"])
    scaler = d["scaler"]
    train_x = scaler.inverse(d["train"][0])
    train_y = scaler.inverse(d["train"][1][:, -1])
    val_x = scaler.inverse(d["val"][0])
    val_y = scaler.inverse(d["val"][1][:, -1])
    test_x = scaler.inverse(d["test"][0])
    test_y = scaler.inverse(d["test"][1][:, -1])
    # Training-only scale for each flow; range scaling aligns training loss with
    # the common per-OD normalized evaluation metric.
    train_tm = d["tm"][:d["t_end"]]
    lo, hi = train_tm.min(axis=0), train_tm.max(axis=0)
    if args.scaling == "mean":
        scale = np.maximum(train_tm.mean(axis=0), 1.0)
    elif args.scaling == "range":
        scale = np.maximum(hi - lo, 1.0)
    else:
        scale = np.maximum(train_tm.std(axis=0), 1.0)
    scale = scale.astype(np.float64)
    mask = ~np.eye(args.nodes, dtype=bool) if args.mask_diag else None

    candidates = []
    if args.per_flow_selection:
        best_pair_score = np.full((args.nodes, args.nodes), np.inf)
        best_pair_pred = np.zeros_like(val_y, dtype=np.float64)
        best_pair_features = np.zeros((args.nodes, args.nodes), dtype=np.int16)
        best_pair_alpha = np.zeros((args.nodes, args.nodes), dtype=np.float64)
    for features in sorted(set([1, 3, 6, 9, args.window])):
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            val_pred, _, _ = _fit_predict(train_x, train_y, val_x, scale,
                                          args.nodes, features, alpha)
            raw = metrics(val_pred, val_y, mask)
            normalized = norm_metrics(val_pred, val_y, lo, hi, mask)
            key = {"raw_rmse": raw["RMSE"], "raw_mae": raw["MAE"],
                   "normalized_rmse": normalized["RMSE"],
                   "normalized_mae": normalized["MAE"]}[args.selection]
            candidates.append((key, features, alpha, val_pred))
            if args.per_flow_selection:
                pair_scale = np.maximum(hi - lo, 1.0)
                pair_score = np.sqrt(np.mean(((val_pred - val_y) / pair_scale) ** 2, axis=0))
                better = pair_score < best_pair_score
                best_pair_score[better] = pair_score[better]
                best_pair_pred[:, better] = val_pred[:, better]
                best_pair_features[better] = features
                best_pair_alpha[better] = alpha
    if args.per_flow_selection:
        pred_val = best_pair_pred
        val_score = norm_metrics(pred_val, val_y, lo, hi, mask)["RMSE"]
        features = int(np.median(best_pair_features[~np.eye(args.nodes, dtype=bool)]))
        alpha = float(np.median(best_pair_alpha[~np.eye(args.nodes, dtype=bool)]))
        selection = {"method": "per-flow validation normalized RMSE",
                     "median_features": features, "median_alpha": alpha,
                     "validation_normalized_rmse": val_score}
    else:
        val_score, features, alpha, pred_val = min(candidates, key=lambda x: x[0])
        selection = {"method": args.selection, "features": features,
                     "alpha": alpha, "validation_score": val_score}
    # Final evaluation uses training+validation windows, but retains the
    # validation-selected hyperparameters. Test targets are not involved in fitting.
    fit_x = np.concatenate([train_x, val_x])
    fit_y = np.concatenate([train_y, val_y])
    if args.per_flow_selection:
        pred_test = _fit_predict_selected(fit_x, fit_y, test_x, scale,
                                          best_pair_features, best_pair_alpha)
    else:
        pred_test, coef, intercept = _fit_predict(fit_x, fit_y, test_x, scale,
                                                  args.nodes, features, alpha)

    def score(pred, truth):
        out = {"original_units": metrics(pred, truth, mask),
               "per_pair_normalized": norm_metrics(pred, truth, lo, hi, mask)}
        return out

    naive_val = np.maximum(val_x[:, -1], 0.0)
    naive_test = np.maximum(test_x[:, -1], 0.0)
    if mask is not None:
        naive_val[:, np.arange(args.nodes), np.arange(args.nodes)] = 0.0
        naive_test[:, np.arange(args.nodes), np.arange(args.nodes)] = 0.0
    report = {
        "flags": vars(args),
        "selected_on_validation": {**selection, "scale": args.scaling,
                                    "per_flow": args.per_flow_selection},
        "validation": {"naive_last_value": score(naive_val, val_y),
                       "ridge": score(pred_val, val_y)},
        "test": {"naive_last_value": score(naive_test, test_y),
                 "ridge": score(pred_test, test_y)},
        "validation_candidate_count": len(candidates),
    }
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/pred_val.npy", pred_val)
    np.save(f"{args.out}/true_val.npy", val_y)
    np.save(f"{args.out}/last_observed_val.npy", val_x[:, -1])
    np.save(f"{args.out}/pred_test.npy", pred_test)
    np.save(f"{args.out}/true_test.npy", test_y)
    np.save(f"{args.out}/last_observed_test.npy", test_x[:, -1])
    if args.per_flow_selection:
        np.savez_compressed(f"{args.out}/model.npz", scale=scale,
                            features=best_pair_features, alpha=best_pair_alpha)
    else:
        np.savez_compressed(f"{args.out}/model.npz", coef=coef, intercept=intercept,
                            scale=scale, features=features, alpha=alpha)
    with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved validation-selected model and forecasts to {args.out}/")


if __name__ == "__main__":
    main()
