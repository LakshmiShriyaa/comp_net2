"""Walk-forward per-OD ridge with delayed-label online updates."""
import json
import os
import numpy as np
from data import load_tm, despike, prepare, metrics, norm_metrics


def feat(X, k, ii, jj, scale):
    lag = X[:, -k:, ii, jj].transpose(0, 2, 1) / scale[None, :, None]
    return np.concatenate([lag, np.ones((*lag.shape[:2], 1))], axis=2)


def target_delta(X, Y, ii, jj, scale):
    return (Y[:, ii, jj] - X[:, -1, ii, jj]) / scale[None]


def fit_initial(F, y, alpha):
    gram = np.einsum("sfk,sfj->fkj", F, F)
    rhs = np.einsum("sfk,sf->fk", F, y)
    dim = F.shape[-1]
    reg = np.diag([alpha] * (dim - 1) + [0.0])
    P = np.linalg.inv(gram + reg[None])
    beta = np.einsum("fkj,fj->fk", P, rhs)
    return beta, P


def update(beta, P, x, y, forgetting):
    px = np.einsum("fij,fj->fi", P, x)
    denom = forgetting + np.einsum("fi,fi->f", x, px)
    gain = px / denom[:, None]
    error = y - np.einsum("fi,fi->f", x, beta)
    beta = beta + gain * error[:, None]
    P = (P - gain[:, :, None] * px[:, None, :]) / forgetting
    return beta, P


def prequential(F_init, y_init, F_online, y_online, X_online, params, start_indices,
                window, horizon, n, pair_i, pair_j, initial_train=True):
    k, alpha, forgetting, scale = params
    beta, P = fit_initial(F_init, y_init, alpha)
    out = np.zeros((len(X_online), len(pair_i)), dtype=np.float64)
    pointer = 0
    for t in range(len(X_online)):
        origin = int(start_indices[t] + window - 1)
        # Use a past online label only when its H-step target is already observed.
        while pointer < len(X_online) and int(start_indices[pointer] + window + horizon - 1) <= origin:
            beta, P = update(beta, P, F_online[pointer], y_online[pointer], forgetting)
            pointer += 1
        delta = np.einsum("fk,fk->f", F_online[t], beta)
        out[t] = np.maximum(X_online[t, -1, pair_i, pair_j] + delta * params[3], 0)
    return out


def main():
    n, w, h, step = 23, 12, 4, 15
    tm, _ = despike(load_tm("geant_final.npy", n))
    times = np.load("geant_times.npy")
    d = prepare(tm, w, h, mask_diag=True, times=times, step_min=step, season=96)
    sc = d["scaler"]
    Xtr, Ytr = sc.inverse(d["train"][0]), sc.inverse(d["train"][1][:, -1])
    Xv, Yv = sc.inverse(d["val"][0]), sc.inverse(d["val"][1][:, -1])
    Xt, Yt = sc.inverse(d["test"][0]), sc.inverse(d["test"][1][:, -1])
    ii, jj = np.where(~np.eye(n, dtype=bool))
    tr = d["tm"][:d["t_end"]]
    lo, hi = tr.min(0), tr.max(0)
    scale2 = np.maximum(hi - lo, 1.0)
    scale = scale2[ii, jj]
    mask = ~np.eye(n, dtype=bool)
    val_params = []
    candidates = []
    for k in (3, 6, 12):
        Ftr = feat(Xtr, k, ii, jj, scale)
        ytr = target_delta(Xtr, Ytr, ii, jj, scale)
        Fv = feat(Xv, k, ii, jj, scale)
        yv = target_delta(Xv, Yv, ii, jj, scale)
        for alpha in (0.1, 1.0, 10.0, 100.0):
            for forgetting in (0.98, 0.995, 1.0):
                params = (k, alpha, forgetting, scale)
                pred = prequential(Ftr, ytr, Fv, yv, Xv, params,
                                   d["val_s"], w, h, n, ii, jj)
                pv = np.zeros((len(Xv), n, n), dtype=np.float64)
                pv[:, ii, jj] = pred
                score = norm_metrics(pv, Yv, lo, hi, mask)["RMSE"]
                candidates.append((score, k, alpha, forgetting, pv))
                print(f"validation k={k} alpha={alpha:g} forgetting={forgetting:g} normalized_rmse={score:.6f}")
    score, k, alpha, forgetting, pred_val = min(candidates, key=lambda x: x[0])

    # Test starts with all train/validation labels, then updates online with
    # previous test targets only after their one-hour label has been observed.
    Xfit, Yfit = np.concatenate([Xtr, Xv]), np.concatenate([Ytr, Yv])
    Ffit = feat(Xfit, k, ii, jj, scale)
    yfit = target_delta(Xfit, Yfit, ii, jj, scale)
    Ft = feat(Xt, k, ii, jj, scale)
    yt = target_delta(Xt, Yt, ii, jj, scale)
    params = (k, alpha, forgetting, scale)
    pred_flat = prequential(Ffit, yfit, Ft, yt, Xt, params,
                            d["test_s"], w, h, n, ii, jj)
    pred_test = np.zeros((len(Xt), n, n), dtype=np.float64)
    pred_test[:, ii, jj] = pred_flat
    last_v, last_t = Xv[:, -1].copy(), Xt[:, -1].copy()
    for p in (pred_val, pred_test, last_v, last_t):
        p[:, np.arange(n), np.arange(n)] = 0
    report = {"selection": {"params": {"lags": k, "alpha": alpha, "forgetting_factor": forgetting},
              "validation_normalized_rmse": score,
              "method": "per-OD ridge updated only after its previous H-step targets are observed"},
              "validation": {"persistence": {"original_units": metrics(last_v,Yv,mask),"normalized": norm_metrics(last_v,Yv,lo,hi,mask)},
                             "online_ridge": {"original_units": metrics(pred_val,Yv,mask),"normalized": norm_metrics(pred_val,Yv,lo,hi,mask)}},
              "test": {"persistence": {"original_units": metrics(last_t,Yt,mask),"normalized": norm_metrics(last_t,Yt,lo,hi,mask)},
                       "online_ridge": {"original_units": metrics(pred_test,Yt,mask),"normalized": norm_metrics(pred_test,Yt,lo,hi,mask)}}}
    out = "res_ONLINE_RIDGE"
    os.makedirs(out, exist_ok=True)
    for name, p in (("pred_val",pred_val),("true_val",Yv),("last_observed_val",last_v),
                    ("pred_test",pred_test),("true_test",Yt),("last_observed_test",last_t)):
        np.save(f"{out}/{name}.npy",p)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
