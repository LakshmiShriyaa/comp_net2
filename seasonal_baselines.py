"""Build and score causal calendar-naive candidates on validation only."""
import json
import os
import numpy as np
from data import load_tm, despike, prepare, metrics, norm_metrics


def main():
    data, times, n, w, h, step = "geant_final.npy", "geant_times.npy", 23, 12, 4, 15
    tm, _ = despike(load_tm(data, n))
    ts = np.load(times)
    d = prepare(tm, w, h, mask_diag=True, times=ts, step_min=step, season=96)
    yv = d["scaler"].inverse(d["val"][1][:, -1])
    xv = d["scaler"].inverse(d["val"][0])
    last = xv[:, -1].copy()
    last[:, np.arange(n), np.arange(n)] = 0
    base = d["t_end"] + d["val_s"] + w + h - 1
    mask = ~np.eye(n, dtype=bool)
    train = d["tm"][:d["t_end"]]
    lo, hi = train.min(axis=0), train.max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    preds = {"persistence": last,
             "moving_average_3": xv[:, -3:].mean(axis=1)}
    for period, name in ((96, "daily_naive"), (672, "weekly_naive")):
        src = base - period
        valid = src >= 0
        valid &= ts[base] - ts[np.maximum(src, 0)] == period * step
        p = last.copy()
        p[valid] = d["tm"][src[valid]]
        p[:, np.arange(n), np.arange(n)] = 0
        preds[name] = p
    both_valid = np.zeros(len(base), dtype=bool)
    daily = preds["daily_naive"]
    weekly = preds["weekly_naive"]
    mean = (daily + weekly) / 2
    # For a missing calendar lookup, that candidate already falls back to persistence.
    preds["daily_weekly_mean"] = mean
    report = {name: {"original_units": metrics(pred, yv, mask),
                     "per_pair_normalized": norm_metrics(pred, yv, lo, hi, mask),
                     "normalized_rmse_improvement_vs_persistence_%":
                         100 * (1 - norm_metrics(pred, yv, lo, hi, mask)["RMSE"] /
                                norm_metrics(last, yv, lo, hi, mask)["RMSE"])}
              for name, pred in preds.items()}
    out = "res_SEASONAL"
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/true_val.npy", yv)
    for name, pred in preds.items():
        np.save(f"{out}/{name}_val.npy", pred)
    with open(f"{out}/validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
