"""Validation-only probes for seasonal and calendar-aware corrections."""
import json
import numpy as np
from data import load_tm, despike
from meta_stack import fit_ridge_stack, apply_stack, score_error


def main():
    n, w, h, step, season = 23, 12, 4, 15, 96
    pv = [np.mean([np.load(f"res_S4L{s}/pred_val_all.npy")[:, -1]
                   for s in ("", "_s1", "_s2")], axis=0),
          np.load("res_NORM_CNN/pred_val_all.npy")[:, -1],
          np.load("res_PERFLOW_MEAN/pred_val.npy"),
          np.load("res_LINEAR_SEASONAL/pred_val.npy"),
          np.load("res_LOG_FLOW/pred_val.npy"),
          np.mean([np.load(f"res_TREE_{s}/pred_val.npy") for s in ("RANGE", "FULL")], axis=0),
          np.load("res_S4L/history_val.npy")[:, -3:].mean(1)]
    base = np.load("res_PERFLOW_MEAN/last_observed_val.npy")
    y = np.load("res_PERFLOW_MEAN/true_val.npy")
    times = np.load("geant_times.npy")
    tm, _ = despike(load_tm("geant_final.npy", n))
    train = tm[:int(len(tm)*.7)]
    scale = np.maximum(train.max(0)-train.min(0), 1).reshape(-1)
    d_val_s = np.load("res_PERFLOW_MEAN/metrics.json", allow_pickle=False) if False else None
    # Recover the exact retained validation target indices with the same window rules.
    from data import prepare
    d = prepare(tm, w, h, mask_diag=True, times=times, step_min=step, season=season)
    target_idx = d["t_end"] + d["val_s"] + w + h - 1
    def seasonal(lag):
        src_time = times[target_idx] - lag*step
        src = np.searchsorted(times, src_time)
        ok = (src < len(times))
        ok &= times[np.minimum(src, len(times)-1)] == src_time
        p = base.copy()
        p[ok] = tm[src[ok]]
        return p
    pv += [seasonal(season), seasonal(7*season)]
    mask = ~np.eye(n, dtype=bool)
    xbase = np.stack([(p-base)[:, mask]/scale[mask.ravel()] for p in pv], axis=1)
    yd = ((y-base)[:, mask]/scale[mask.ravel()]).astype(float)
    minute = times[target_idx]
    phase = 2*np.pi*(minute % 1440)/1440
    features = [np.sin(phase*k) for k in range(1,5)] + [np.cos(phase*k) for k in range(1,5)]
    week = 2*np.pi*((minute//1440)%7)/7
    features += [np.sin(week), np.cos(week)]
    cal = np.stack(features, axis=1)
    xcal = np.broadcast_to(cal[:, :, None], (len(cal), cal.shape[1], yd.shape[1]))
    folds=[(800,1200),(1200,len(y))]
    alphas=(.001,.01,.1,1,10,100,1000)
    results={}
    for tag,x in [("base_plus_seasonal",xbase), ("base_seasonal_plus_calendar",np.concatenate([xbase,xcal],axis=1))]:
        scores={a:[] for a in alphas}
        for start,end in folds:
            for a in alphas:
                b,c=fit_ridge_stack(x[:start],yd[:start],a)
                scores[a].append(score_error(apply_stack(x[start:end],b,c),yd[start:end]))
        best=min(scores,key=lambda a:np.mean(scores[a]))
        results[tag]={"best_alpha":best,"fold_scores":scores[best],"mean":float(np.mean(scores[best]))}
    baseline=float(np.mean([score_error(np.zeros_like(yd[s:e]),yd[s:e]) for s,e in folds]))
    results["persistence_cv_rmse"]=baseline
    for tag in ("base_plus_seasonal","base_seasonal_plus_calendar"):
        results[tag]["improvement_vs_persistence_percent"]=100*(1-results[tag]["mean"]/baseline)
    print(json.dumps(results,indent=2))

if __name__ == "__main__": main()
