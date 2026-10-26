"""Per-OD locally extrapolated trends, with damping selected on validation."""
import json
import os
import numpy as np
from data import load_tm, despike, prepare, metrics, norm_metrics


def predict(X, k, damp, horizon):
    x = X[:, -k:]
    t = np.arange(k, dtype=np.float64)
    centered = t - t.mean()
    slope = np.einsum("w,swij->sij", centered, x) / np.sum(centered ** 2)
    return np.maximum(x[:, -1] + slope * horizon * damp, 0)


def main():
    n, w, h = 23, 12, 4
    tm, _ = despike(load_tm("geant_final.npy", n))
    d = prepare(tm, w, h, mask_diag=True, times=np.load("geant_times.npy"), step_min=15, season=96)
    Xv, Yv = d["scaler"].inverse(d["val"][0]), d["scaler"].inverse(d["val"][1][:, -1])
    Xt, Yt = d["scaler"].inverse(d["test"][0]), d["scaler"].inverse(d["test"][1][:, -1])
    last_v, last_t = Xv[:, -1].copy(), Xt[:, -1].copy()
    nmask = ~np.eye(n, dtype=bool)
    tr = d["tm"][:d["t_end"]]
    lo, hi = tr.min(0), tr.max(0)
    candidates = [(k, damp) for k in (3, 6, 9, 12) for damp in (0, .2, .4, .6, .8, 1.0)]
    best = np.full((n, n), np.inf)
    chosen_k = np.zeros((n, n), dtype=int)
    chosen_damp = np.zeros((n, n), dtype=float)
    pv_best = np.zeros_like(Yv, dtype=np.float64)
    score_grid = {}
    for k, damp in candidates:
        pv = predict(Xv, k, damp, h)
        loss = (((pv - Yv) / np.maximum(hi - lo, 1)) ** 2).mean(0)
        score_grid[f"k{k}_d{damp}"] = norm_metrics(pv, Yv, lo, hi, nmask)["RMSE"]
        use = loss < best
        best[use], chosen_k[use], chosen_damp[use] = loss[use], k, damp
        pv_best[:, use] = pv[:, use]
    pt = np.zeros_like(Yt, dtype=np.float64)
    for k, damp in candidates:
        use = (chosen_k == k) & (chosen_damp == damp) & nmask
        if use.any():
            pt[:, use] = predict(Xt, k, damp, h)[:, use]
    for p in (pv_best, pt, last_v, last_t):
        p[:, np.arange(n), np.arange(n)] = 0
    report = {"selection": "per-OD history length and trend damping selected on validation normalized RMSE",
              "selected_configs": {"history_length_counts": {str(k): int((chosen_k[nmask] == k).sum()) for k in (3,6,9,12)},
                                   "damping_mean": float(chosen_damp[nmask].mean()),
                                   "validation_candidate_scores": score_grid},
              "validation": {"persistence": {"original_units": metrics(last_v,Yv,nmask),"normalized": norm_metrics(last_v,Yv,lo,hi,nmask)},
                             "local_trend": {"original_units": metrics(pv_best,Yv,nmask),"normalized": norm_metrics(pv_best,Yv,lo,hi,nmask)}},
              "test": {"persistence": {"original_units": metrics(last_t,Yt,nmask),"normalized": norm_metrics(last_t,Yt,lo,hi,nmask)},
                       "local_trend": {"original_units": metrics(pt,Yt,nmask),"normalized": norm_metrics(pt,Yt,lo,hi,nmask)}}}
    out = "res_TREND"
    os.makedirs(out, exist_ok=True)
    for name,p in (("pred_val",pv_best),("true_val",Yv),("last_observed_val",last_v),
                   ("pred_test",pt),("true_test",Yt),("last_observed_test",last_t)):
        np.save(f"{out}/{name}.npy",p)
    np.savez_compressed(f"{out}/params.npz",history=chosen_k,damping=chosen_damp)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
