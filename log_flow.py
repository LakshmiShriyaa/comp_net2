"""Per-OD log-space autoregression with seasonal references and validation selection."""
import json
import os
import numpy as np
from data import load_tm, despike, prepare, metrics, norm_metrics


def make_split(d, times, split, start, window, horizon, scale, n, step_min):
    X = d[split][0]
    X = d["scaler"].inverse(X)
    Y = d["scaler"].inverse(d[split][1][:, -1])
    target = start + d[split + "_s"] + window + horizon - 1
    xnorm = np.maximum(X, 0) / scale[None, None]
    blocks = [np.log1p(xnorm)]
    for lag in (96, 672):
        src = target - lag
        valid = src >= 0
        valid &= times[target] - times[np.maximum(src, 0)] == lag * step_min
        values = X[:, -1].copy()
        values[valid] = d["tm"][src[valid]]
        blocks.append(np.log1p(np.maximum(values, 0) / scale[None]))
    minute = times[target]
    day = 2 * np.pi * (minute % 1440) / 1440
    week = 2 * np.pi * ((minute // 1440) % 7) / 7
    cyc = np.stack([np.sin(day), np.cos(day), np.sin(week), np.cos(week)], axis=1)
    cyc = np.broadcast_to(cyc[:, :, None, None], (len(X), 4, n, n))
    F = np.concatenate([blocks[0], blocks[1][:, None], blocks[2][:, None], cyc], axis=1)
    ylog = np.log1p(np.maximum(Y, 0) / scale[None])
    return F.transpose(0, 1, 2, 3), X, Y, ylog


def fit_one(x, y, alpha, xe):
    xm, ym = x.mean(0), y.mean()
    xc, yc = x - xm, y - ym
    beta = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    return (xe - xm) @ beta + ym


def main():
    n, w, h, step = 23, 12, 4, 15
    tm, _ = despike(load_tm("geant_final.npy", n))
    times = np.load("geant_times.npy")
    d = prepare(tm, w, h, mask_diag=True, times=times, step_min=step, season=96)
    train_tm = d["tm"][:d["t_end"]]
    scale = np.maximum(train_tm.max(0) - train_tm.min(0), 1.0)
    parts = {"train": make_split(d, times, "train", 0, w, h, scale, n, step),
             "val": make_split(d, times, "val", d["t_end"], w, h, scale, n, step),
             "test": make_split(d, times, "test", d["v_end"], w, h, scale, n, step)}
    Ftr, Xtr, Ytr, Ltr = parts["train"]
    Fv, Xv, Yv, Lv = parts["val"]
    Ft, Xt, Yt, Lt = parts["test"]
    mask = ~np.eye(n, dtype=bool)
    lo, hi = train_tm.min(0), train_tm.max(0)
    alphas = (0.001, 0.01, 0.1, 1, 10, 100)
    best = np.full((n, n), np.inf)
    best_a = np.zeros((n, n))
    pred_v = np.zeros_like(Yv, dtype=np.float64)
    for alpha in alphas:
        current = np.zeros_like(Yv, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                z = fit_one(Ftr[:, :, i, j], Ltr[:, i, j], alpha, Fv[:, :, i, j])
                current[:, i, j] = np.maximum(np.expm1(np.clip(z, -20, 20)) * scale[i, j], 0)
        loss = (((current - Yv) / np.maximum(hi - lo, 1.0)) ** 2).mean(0)
        update = loss < best
        best[update], best_a[update], pred_v[:, update] = loss[update], alpha, current[:, update]
        print(f"validation alpha={alpha:g} normalized_rmse={norm_metrics(current,Yv,lo,hi,mask)['RMSE']:.6f}")
    Ffit, Lfit = np.concatenate([Ftr, Fv]), np.concatenate([Ltr, Lv])
    pred_t = np.zeros_like(Yt, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            z = fit_one(Ffit[:, :, i, j], Lfit[:, i, j], best_a[i, j], Ft[:, :, i, j])
            pred_t[:, i, j] = np.maximum(np.expm1(np.clip(z, -20, 20)) * scale[i, j], 0)
    last_v, last_t = Xv[:, -1].copy(), Xt[:, -1].copy()
    for p in (pred_v, pred_t, last_v, last_t):
        p[:, np.arange(n), np.arange(n)] = 0
    report = {"selection": "per-OD alpha chosen on validation normalized RMSE",
              "alpha_counts": {str(a): int((best_a[mask] == a).sum()) for a in alphas},
              "validation": {"persistence": {"original_units": metrics(last_v,Yv,mask),"normalized": norm_metrics(last_v,Yv,lo,hi,mask)},
                             "log_flow": {"original_units": metrics(pred_v,Yv,mask),"normalized": norm_metrics(pred_v,Yv,lo,hi,mask)}},
              "test": {"persistence": {"original_units": metrics(last_t,Yt,mask),"normalized": norm_metrics(last_t,Yt,lo,hi,mask)},
                       "log_flow": {"original_units": metrics(pred_t,Yt,mask),"normalized": norm_metrics(pred_t,Yt,lo,hi,mask)}}}
    out = "res_LOG_FLOW"
    os.makedirs(out, exist_ok=True)
    for name, p in (("pred_val",pred_v),("true_val",Yv),("last_observed_val",last_v),
                    ("pred_test",pred_t),("true_test",Yt),("last_observed_test",last_t)):
        np.save(f"{out}/{name}.npy",p)
    np.save(f"{out}/alpha_by_od.npy",best_a)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
