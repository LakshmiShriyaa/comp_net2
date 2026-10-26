"""Causal Butterworth low-pass traffic forecasts, selected on validation."""
import json
import os
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
from data import load_tm, despike, prepare, metrics, norm_metrics


def causal_filter(tm, times, cutoff, step_min):
    sos = butter(2, cutoff, btype="lowpass", output="sos")
    out = np.zeros_like(tm, dtype=np.float64)
    breaks = np.r_[0, np.where(np.diff(times) != step_min)[0] + 1, len(times)]
    for a, b in zip(breaks[:-1], breaks[1:]):
        x = tm[a:b]
        zi = sosfilt_zi(sos)[:, :, None, None] * x[0][None, ...]
        out[a:b], _ = sosfilt(sos, x, axis=0, zi=zi)
    return np.maximum(out, 0)


def main():
    n, w, h, step = 23, 12, 4, 15
    tm, _ = despike(load_tm("geant_final.npy", n))
    tm = tm.copy(); ix = np.arange(n); tm[:, ix, ix] = 0
    times = np.load("geant_times.npy")
    d = prepare(tm, w, h, mask_diag=True, times=times, step_min=step, season=96)
    lo, hi = d["tm"][:d["t_end"]].min(0), d["tm"][:d["t_end"]].max(0)
    mask = ~np.eye(n, dtype=bool)
    candidates = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    pred_by_cutoff = {}
    scores = {}
    for cutoff in candidates:
        filt = causal_filter(d["tm"], times, cutoff, step)
        preds = {}
        for name in ("val", "test"):
            start = {"val": d["t_end"], "test": d["v_end"]}[name]
            issue_idx = start + d[name + "_s"] + w - 1
            preds[name] = filt[issue_idx]
        pred_by_cutoff[cutoff] = preds
        yv = d["scaler"].inverse(d["val"][1][:, -1])
        scores[cutoff] = {"original_units": metrics(preds["val"], yv, mask),
                          "normalized": norm_metrics(preds["val"], yv, lo, hi, mask)}
        print(f"validation cutoff={cutoff:g} normalized_rmse={scores[cutoff]['normalized']['RMSE']:.6f}")
    best = min(candidates, key=lambda c: scores[c]["normalized"]["RMSE"])
    # Score only the validation-selected causal filter on the test period.
    out = "res_BUTTERWORTH"
    os.makedirs(out, exist_ok=True)
    yv = d["scaler"].inverse(d["val"][1][:, -1])
    yt = d["scaler"].inverse(d["test"][1][:, -1])
    last_v = d["scaler"].inverse(d["val"][0][:, -1]); last_t = d["scaler"].inverse(d["test"][0][:, -1])
    for p in (yv, yt, last_v, last_t, pred_by_cutoff[best]["val"], pred_by_cutoff[best]["test"]):
        p[:, ix, ix] = 0
    report = {"selected_on_validation": {"cutoff": best, "score_grid": scores},
              "validation": {"persistence": {"original_units": metrics(last_v,yv,mask),"normalized": norm_metrics(last_v,yv,lo,hi,mask)},
                             "butterworth": {"original_units": metrics(pred_by_cutoff[best]["val"],yv,mask),"normalized": norm_metrics(pred_by_cutoff[best]["val"],yv,lo,hi,mask)}},
              "test": {"persistence": {"original_units": metrics(last_t,yt,mask),"normalized": norm_metrics(last_t,yt,lo,hi,mask)},
                       "butterworth": {"original_units": metrics(pred_by_cutoff[best]["test"],yt,mask),"normalized": norm_metrics(pred_by_cutoff[best]["test"],yt,lo,hi,mask)}}}
    for name, p in (("pred_val",pred_by_cutoff[best]["val"]),("true_val",yv),("last_observed_val",last_v),
                    ("pred_test",pred_by_cutoff[best]["test"]),("true_test",yt),("last_observed_test",last_t)):
        np.save(f"{out}/{name}.npy",p)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
