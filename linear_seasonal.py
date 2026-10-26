"""Per-OD ridge using recent traffic, seasonal lags, and calendar phase features."""
import argparse
import json
import os
import numpy as np
from data import load_tm, despike, prepare, metrics, norm_metrics


def build_features(d, times, window, horizon, step_min, periods, n):
    arrays = {}
    starts = {"train": 0, "val": d["t_end"], "test": d["v_end"]}
    for name in ("train", "val", "test"):
        X = d["scaler"].inverse(d[name][0])
        Y = d["scaler"].inverse(d[name][1][:, -1])
        target_idx = starts[name] + d[name + "_s"] + window + horizon - 1
        blocks = [X / periods["scale"]]
        for lag in (96, 672):
            src = target_idx - lag
            valid = src >= 0
            valid &= times[target_idx] - times[np.maximum(src, 0)] == lag * step_min
            seasonal = X[:, -1].copy()
            seasonal[valid] = d["tm"][src[valid]]
            blocks.append(seasonal[:, None] / periods["scale"])
        minute = times[target_idx]
        day = 2 * np.pi * (minute % 1440) / 1440
        week = 2 * np.pi * ((minute // 1440) % 7) / 7
        cyc = np.stack([np.sin(day), np.cos(day), np.sin(week), np.cos(week)], axis=1)
        # Recent traffic levels, two seasonal reference values, and clock phase.
        F = np.concatenate([blocks[0], blocks[1], blocks[2], cyc[:, :, None, None]
                            .repeat(n, axis=2).repeat(n, axis=3)], axis=1)
        arrays[name] = (F.astype(np.float64), X, Y)
    return arrays


def fit_predict(Ftr, Xtr, Ytr, Fe, scale, alpha, n):
    pred = np.zeros((len(Fe), n, n), dtype=np.float64)
    coeff = np.zeros((n, n, Ftr.shape[1]), dtype=np.float64)
    bias = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x, xe = Ftr[:, :, i, j], Fe[:, :, i, j]
            y = (Ytr[:, i, j] - Xtr[:, -1, i, j]) / scale[i, j]
            xm, ym = x.mean(axis=0), y.mean()
            xc, yc = x - xm, y - ym
            beta = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
            b = ym - xm @ beta
            pred[:, i, j] = np.maximum(Xeval_last[0][:, i, j] + (xe @ beta + b) * scale[i, j], 0)
            coeff[i, j], bias[i, j] = beta, b
    return pred, coeff, bias


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="geant_final.npy")
    ap.add_argument("--times", default="geant_times.npy")
    ap.add_argument("--out", default="res_LINEAR_SEASONAL")
    ap.add_argument("--nodes", type=int, default=23)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--step-min", type=int, default=15)
    args = ap.parse_args()
    n = args.nodes
    tm, _ = despike(load_tm(args.data, n))
    times = np.load(args.times)
    d = prepare(tm, args.window, args.horizon, mask_diag=True, times=times,
                step_min=args.step_min, season=96)
    train_tm = d["tm"][:d["t_end"]]
    scale = np.maximum(train_tm.max(axis=0) - train_tm.min(axis=0), 1.0)
    arr = build_features(d, times, args.window, args.horizon, args.step_min,
                         {"scale": scale}, n)
    Ftr, Xtr, Ytr = arr["train"]
    Fv, Xv, Yv = arr["val"]
    Ft, Xt, Yt = arr["test"]
    mask = ~np.eye(n, dtype=bool)
    lo, hi = train_tm.min(axis=0), train_tm.max(axis=0)

    # Validation picks a separate regularization strength for every OD pair.
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)
    best_loss = np.full((n, n), np.inf)
    best_alpha = np.zeros((n, n), dtype=np.float64)
    best_val = np.zeros_like(Yv, dtype=np.float64)
    for alpha in alphas:
        pv = np.zeros_like(Yv, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x = Ftr[:, :, i, j]
                y = (Ytr[:, i, j] - Xtr[:, -1, i, j]) / scale[i, j]
                xm, ym = x.mean(axis=0), y.mean()
                xc, yc = x - xm, y - ym
                beta = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
                b = ym - xm @ beta
                pv[:, i, j] = np.maximum(Xv[:, -1, i, j] +
                    (Fv[:, :, i, j] @ beta + b) * scale[i, j], 0)
        pair_loss = (((pv - Yv) / np.maximum(hi - lo, 1.0)) ** 2).mean(axis=0)
        better = pair_loss < best_loss
        best_loss[better], best_alpha[better], best_val[:, better] = pair_loss[better], alpha, pv[:, better]
        print(f"validation alpha={alpha:g} pooled_normalized_rmse={norm_metrics(pv, Yv, lo, hi, mask)['RMSE']:.6f}")

    # Refit selected per-flow models on train+validation windows, then test once.
    Ffit, Xfit, Yfit = np.concatenate([Ftr, Fv]), np.concatenate([Xtr, Xv]), np.concatenate([Ytr, Yv])
    Xeval_last = (Xt[:, -1],)
    pred_test = np.zeros_like(Yt, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x, xe = Ffit[:, :, i, j], Ft[:, :, i, j]
            y = (Yfit[:, i, j] - Xfit[:, -1, i, j]) / scale[i, j]
            xm, ym = x.mean(axis=0), y.mean()
            xc, yc = x - xm, y - ym
            alpha = best_alpha[i, j]
            beta = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
            b = ym - xm @ beta
            pred_test[:, i, j] = np.maximum(Xt[:, -1, i, j] + (xe @ beta + b) * scale[i, j], 0)
    last_val, last_test = Xv[:, -1].copy(), Xt[:, -1].copy()
    for p in (last_val, last_test, best_val, pred_test):
        p[:, np.arange(n), np.arange(n)] = 0
    report = {
        "selection": "per-OD alpha selected from validation normalized squared error",
        "alphas": {str(a): int((best_alpha[mask] == a).sum()) for a in alphas},
        "validation": {"persistence": {"original_units": metrics(last_val, Yv, mask), "normalized": norm_metrics(last_val, Yv, lo, hi, mask)},
                       "seasonal_ridge": {"original_units": metrics(best_val, Yv, mask), "normalized": norm_metrics(best_val, Yv, lo, hi, mask)}},
        "test": {"persistence": {"original_units": metrics(last_test, Yt, mask), "normalized": norm_metrics(last_test, Yt, lo, hi, mask)},
                 "seasonal_ridge": {"original_units": metrics(pred_test, Yt, mask), "normalized": norm_metrics(pred_test, Yt, lo, hi, mask)}}}
    os.makedirs(args.out, exist_ok=True)
    for name, p in (("pred_val", best_val), ("true_val", Yv), ("last_observed_val", last_val),
                    ("pred_test", pred_test), ("true_test", Yt), ("last_observed_test", last_test)):
        np.save(f"{args.out}/{name}.npy", p)
    np.save(f"{args.out}/alpha_by_od.npy", best_alpha)
    with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # fit_predict is retained as a helper for notebooks; it uses the module-level
    # Xeval_last set during evaluation.
    main()
