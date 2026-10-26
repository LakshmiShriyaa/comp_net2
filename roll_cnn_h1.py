"""Roll the existing one-step shared CNN-LSTM forward four steps."""
import json
import os
import numpy as np
import torch
from data import load_tm, despike, prepare, metrics, norm_metrics
from train import build, predict


def rollout(model, X, horizon):
    hist = X.copy()
    for _ in range(horizon):
        nxt = np.maximum(predict(model, hist, "cpu")[:, 0], 0)
        hist = np.concatenate([hist[:, 1:], nxt[:, None]], axis=1)
    return hist[:, -1]


def main():
    torch.set_num_threads(1)
    n, w, h = 23, 12, 4
    tm, _ = despike(load_tm("geant_final.npy", n))
    times = np.load("geant_times.npy")
    d = prepare(tm, w, h, mask_diag=True, times=times, step_min=15, season=96)
    level = d["scaler"].transform(d["tm"][:d["t_end"]]).mean(0)
    scale = np.maximum(level, 1e-3 * level.max()).astype(np.float32)
    model = build("cnn", n, 1, residual=False, shared=True, scale=scale,
                  local_scale=True, use_tod=False)
    model.load_state_dict(torch.load("res_H1/cnn_lstm.pt", map_location="cpu", weights_only=True))
    val = np.maximum(d["scaler"].inverse(rollout(model, d["val"][0], h)), 0)
    test = np.maximum(d["scaler"].inverse(rollout(model, d["test"][0], h)), 0)
    yv = d["scaler"].inverse(d["val"][1][:, -1])
    yt = d["scaler"].inverse(d["test"][1][:, -1])
    last_v = d["scaler"].inverse(d["val"][0][:, -1])
    last_t = d["scaler"].inverse(d["test"][0][:, -1])
    mask = ~np.eye(n, dtype=bool)
    tr = d["tm"][:d["t_end"]]
    lo, hi = tr.min(0), tr.max(0)
    for p in (val, test, yv, yt, last_v, last_t):
        if p.ndim == 4:
            p[:, :, np.arange(n), np.arange(n)] = 0
        else:
            p[:, np.arange(n), np.arange(n)] = 0
    report = {"model": "existing H1 CNN-LSTM recursively rolled forward four times; fitted checkpoint fixed before this evaluation",
              "validation": {"persistence": {"original_units": metrics(last_v,yv,mask),"normalized": norm_metrics(last_v,yv,lo,hi,mask)},
                             "recursive_cnn": {"original_units": metrics(val,yv,mask),"normalized": norm_metrics(val,yv,lo,hi,mask)}},
              "test": {"persistence": {"original_units": metrics(last_t,yt,mask),"normalized": norm_metrics(last_t,yt,lo,hi,mask)},
                       "recursive_cnn": {"original_units": metrics(test,yt,mask),"normalized": norm_metrics(test,yt,lo,hi,mask)}}}
    out = "res_ROLL_CNN"
    os.makedirs(out, exist_ok=True)
    for name, p in (("pred_val",val),("true_val",yv),("last_observed_val",last_v),
                    ("pred_test",test),("true_test",yt),("last_observed_test",last_t)):
        np.save(f"{out}/{name}.npy",p)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
