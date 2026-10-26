"""Generate validation forecasts from already-trained CNN-LSTM seed checkpoints."""
import os
import numpy as np
import torch
from data import load_tm, despike, prepare
from train import build, predict


def main():
    torch.set_num_threads(1)
    n, window, horizon = 23, 12, 4
    tm, _ = despike(load_tm("geant_final.npy", n))
    times = np.load("geant_times.npy")
    d = prepare(tm, window, horizon, mask_diag=True, times=times,
                step_min=15, season=96)
    sc = d["scaler"]
    level = sc.transform(d["tm"][:d["t_end"]]).mean(axis=0)
    scale = np.maximum(level, 1e-3 * level.max()).astype(np.float32)
    for seed in (1, 2):
        out = f"res_S4L_s{seed}"
        model = build("cnn", n, horizon, residual=False, shared=True,
                      scale=scale, local_scale=True, use_tod=False)
        model.load_state_dict(torch.load(f"{out}/cnn_lstm.pt", map_location="cpu", weights_only=True))
        pv = np.maximum(sc.inverse(predict(model, d["val"][0], "cpu")), 0)
        pt = np.maximum(sc.inverse(predict(model, d["test"][0], "cpu")), 0)
        yv, yt = sc.inverse(d["val"][1]), sc.inverse(d["test"][1])
        last_v = sc.inverse(d["val"][0][:, -1])
        last_t = sc.inverse(d["test"][0][:, -1])
        for p in (pv, pt, yv, yt):
            p[:, :, np.arange(n), np.arange(n)] = 0
        for p in (last_v, last_t):
            p[:, np.arange(n), np.arange(n)] = 0
        np.save(f"{out}/pred_val_all.npy", pv)
        np.save(f"{out}/true_val_all.npy", yv)
        np.save(f"{out}/pred_test_all_reinferred.npy", pt)
        np.save(f"{out}/last_observed_val.npy", last_v)
        np.save(f"{out}/last_observed_test.npy", last_t)
        print(f"seed {seed}: generated validation and test forecasts; test forecast saved for parity check")


if __name__ == "__main__":
    main()
