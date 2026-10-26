"""Validation-selected per-OD one-step ridge rolled forward to a one-hour forecast."""
import json
import os
import numpy as np
from data import load_tm, despike, prepare, metrics, norm_metrics


def fit_models(X, Y1, scale, features, alpha, n):
    models = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            k = int(features[i, j])
            s = scale[i, j]
            x = X[:, -k:, i, j] / s
            y = (Y1[:, i, j] - X[:, -1, i, j]) / s
            xm, ym = x.mean(0), y.mean()
            xc, yc = x - xm, y - ym
            a = float(alpha[i, j])
            beta = np.linalg.solve(xc.T @ xc + a * np.eye(k), xc.T @ yc)
            models[i, j] = (k, s, xm, beta, ym - xm @ beta)
    return models


def rollout(X, models, n, horizon):
    hist = X.copy()
    pred = None
    for _ in range(horizon):
        pred = np.zeros((len(X), n, n), dtype=np.float64)
        for (i, j), (k, s, xm, beta, bias) in models.items():
            x = hist[:, -k:, i, j] / s
            delta = (x @ beta + bias) * s
            pred[:, i, j] = np.maximum(hist[:, -1, i, j] + delta, 0)
        hist = np.concatenate([hist[:, 1:], pred[:, None]], axis=1)
    return pred


def main():
    n, window, horizon = 23, 12, 4
    tm, _ = despike(load_tm("geant_final.npy", n))
    times = np.load("geant_times.npy")
    d = prepare(tm, window, horizon, mask_diag=True, times=times, step_min=15, season=96)
    sc = d["scaler"]
    Xtr, Ytr = sc.inverse(d["train"][0]), sc.inverse(d["train"][1])
    Xv, Yv = sc.inverse(d["val"][0]), sc.inverse(d["val"][1])
    Xt, Yt = sc.inverse(d["test"][0]), sc.inverse(d["test"][1])
    Ytr1 = Ytr[:, 0]
    lo, hi = d["tm"][:d["t_end"]].min(0), d["tm"][:d["t_end"]].max(0)
    scale = np.maximum(hi - lo, 1.0)
    mask = ~np.eye(n, dtype=bool)
    feature_grid = sorted(set([1, 3, 6, 9, window]))
    alpha_grid = (0.1, 1.0, 10.0, 100.0, 1000.0)
    best = np.full((n, n), np.inf)
    best_k = np.zeros((n, n), dtype=np.int16)
    best_a = np.zeros((n, n), dtype=np.float64)
    best_val = np.zeros((len(Xv), n, n), dtype=np.float64)
    for k in feature_grid:
        kk = np.full((n, n), k, dtype=np.int16)
        for alpha in alpha_grid:
            aa = np.full((n, n), alpha, dtype=np.float64)
            models = fit_models(Xtr, Ytr1, scale, kk, aa, n)
            pv = rollout(Xv, models, n, horizon)
            loss = (((pv - Yv[:, -1]) / np.maximum(hi - lo, 1.0)) ** 2).mean(0)
            update = loss < best
            best[update], best_k[update], best_a[update] = loss[update], k, alpha
            best_val[:, update] = pv[:, update]
        print(f"finished candidate lag_count={k}")

    # Refit selected one-step models on all train+validation windows.
    Xfit, Yfit = np.concatenate([Xtr, Xv]), np.concatenate([Ytr, Yv])
    models = fit_models(Xfit, Yfit[:, 0], scale, best_k, best_a, n)
    pt = rollout(Xt, models, n, horizon)
    last_v, last_t = Xv[:, -1].copy(), Xt[:, -1].copy()
    for p in (best_val, pt, last_v, last_t):
        p[:, np.arange(n), np.arange(n)] = 0
    report = {"selection": "per-OD lag count and ridge alpha selected by validation normalized RMSE after four-step recursive rollout",
              "lag_counts": {str(k): int((best_k[mask] == k).sum()) for k in feature_grid},
              "alphas": {str(a): int((best_a[mask] == a).sum()) for a in alpha_grid},
              "validation": {"persistence": {"original_units": metrics(last_v,Yv[:,-1],mask),"normalized": norm_metrics(last_v,Yv[:,-1],lo,hi,mask)},
                             "recursive_ridge": {"original_units": metrics(best_val,Yv[:,-1],mask),"normalized": norm_metrics(best_val,Yv[:,-1],lo,hi,mask)}},
              "test": {"persistence": {"original_units": metrics(last_t,Yt[:,-1],mask),"normalized": norm_metrics(last_t,Yt[:,-1],lo,hi,mask)},
                       "recursive_ridge": {"original_units": metrics(pt,Yt[:,-1],mask),"normalized": norm_metrics(pt,Yt[:,-1],lo,hi,mask)}}}
    out = "res_RECURSIVE_RIDGE"
    os.makedirs(out, exist_ok=True)
    for name, p in (("pred_val",best_val),("true_val",Yv[:,-1]),("last_observed_val",last_v),
                    ("pred_test",pt),("true_test",Yt[:,-1]),("last_observed_test",last_t)):
        np.save(f"{out}/{name}.npy",p)
    np.savez_compressed(f"{out}/params.npz",lag_count=best_k,alpha=best_a,scale=scale)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
