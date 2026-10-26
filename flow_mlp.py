"""Train a shared nonlinear predictor across OD flows.

Each OD pair contributes a normalized history vector, while a single compact MLP
shares its temporal dynamics across all flows. Time-of-day and weekly phase are
known covariates for the future target. Hyperparameters/epoch are selected on
validation only; the test split is evaluated after selection.

Example:
    py flow_mlp.py --data geant_final.npy --nodes 23 --times geant_times.npy \
        --horizon 4 --window 12 --scale range --out res_MLP_RANGE
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import load_tm, despike, prepare, metrics, norm_metrics


class FlowMLP(nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _time_features(times, start, starts, window, horizon):
    target_idx = start + starts + window + horizon - 1
    minutes = np.asarray(times)[target_idx]
    day_phase = 2 * np.pi * (minutes % 1440) / 1440.0
    week_phase = 2 * np.pi * ((minutes // 1440) % 7) / 7.0
    return np.stack([np.sin(day_phase), np.cos(day_phase),
                     np.sin(week_phase), np.cos(week_phase)], axis=1).astype(np.float32)


def _make_xy(X, Y, tod, scale, stride=1):
    if stride > 1:
        X, Y, tod = X[::stride], Y[::stride], tod[::stride]
    s, w, n, _ = X.shape
    flows = n * n
    # Log-relative history handles bursts; residual target remains additive in
    # normalized traffic units so the forecast starts from the last observation.
    history = np.log1p(np.maximum(X, 0) / scale).transpose(0, 2, 3, 1).reshape(s * flows, w)
    time = np.broadcast_to(tod[:, None, :], (s, flows, tod.shape[1])).reshape(s * flows, -1)
    features = np.concatenate([history, time], axis=1).astype(np.float32)
    target = ((Y - X[:, -1]) / scale).reshape(s, flows).reshape(-1).astype(np.float32)
    last = X[:, -1].reshape(s, flows)
    return features, target, last


@torch.no_grad()
def _predict(model, X, last, scale, nodes, device, batch_size):
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    delta = np.concatenate(out).reshape(len(last), nodes * nodes)
    pred = (last + delta * scale.reshape(1, -1)).reshape(len(last), nodes, nodes)
    pred = np.maximum(pred, 0.0)
    pred[:, np.arange(nodes), np.arange(nodes)] = 0.0
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--step-min", type=int, default=15)
    ap.add_argument("--season", type=int, default=96)
    ap.add_argument("--scale", choices=("mean", "range", "std"), default="range")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--train-stride", type=int, default=1,
                    help="subsample training windows at this stride; validation/test stay complete")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tm, fixed = despike(load_tm(args.data, args.nodes))
    times = np.load(args.times)
    d = prepare(tm, args.window, args.horizon, mask_diag=True,
                times=times, step_min=args.step_min, season=args.season)
    print(f"device={device}; repaired={len(fixed)}; windows={d['stats']}")
    train_tm = d["tm"][:d["t_end"]]
    if args.scale == "mean":
        scale = np.maximum(train_tm.mean(axis=0), 1.0)
    elif args.scale == "range":
        scale = np.maximum(train_tm.max(axis=0) - train_tm.min(axis=0), 1.0)
    else:
        scale = np.maximum(train_tm.std(axis=0), 1.0)
    scale = scale.astype(np.float32)

    bounds = {"train": 0, "val": d["t_end"], "test": d["v_end"]}
    split = {}
    n = args.nodes
    mask = ~np.eye(n, dtype=bool)
    true_val = d["scaler"].inverse(d["val"][1][:, -1])
    true_test = d["scaler"].inverse(d["test"][1][:, -1])
    last_val = d["scaler"].inverse(d["val"][0][:, -1])
    last_test = d["scaler"].inverse(d["test"][0][:, -1])
    # The data pipeline returns scaled windows; restore original traffic units
    # before feature generation so the flow-specific scale is applied exactly once.
    Xtr_raw = d["scaler"].inverse(d["train"][0])
    Ytr_raw = d["scaler"].inverse(d["train"][1][:, -1])
    Xva_raw = d["scaler"].inverse(d["val"][0])
    Yva_raw = d["scaler"].inverse(d["val"][1][:, -1])
    Xte_raw = d["scaler"].inverse(d["test"][0])
    Yte_raw = d["scaler"].inverse(d["test"][1][:, -1])
    for name, X, Y in (("train", Xtr_raw, Ytr_raw), ("val", Xva_raw, Yva_raw), ("test", Xte_raw, Yte_raw)):
        tod = _time_features(times, bounds[name], d[name + "_s"], args.window, args.horizon)
        split[name] = _make_xy(X, Y, tod, scale,
                               stride=args.train_stride if name == "train" else 1)
    Xtr, ytr, _ = split["train"]
    Xva, yva, last_va = split["val"]
    Xte, yte, last_te = split["test"]

    model = FlowMLP(Xtr.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                              batch_size=args.batch, shuffle=True)
    xv_t = torch.from_numpy(Xva)
    yv_t = torch.from_numpy(yva)
    best_score, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
        model.eval()
        with torch.no_grad():
            vals = []
            for i in range(0, len(xv_t), args.batch):
                vals.append(model(xv_t[i:i + args.batch].to(device)).cpu().numpy())
        val_delta = np.concatenate(vals)
        val_pred = (last_va + val_delta.reshape(len(last_va), n*n) * scale.reshape(1, -1)).reshape(-1, n, n)
        val_pred = np.maximum(val_pred, 0.0)
        val_pred[:, np.arange(n), np.arange(n)] = 0.0
        val_score = norm_metrics(val_pred, true_val, train_tm.min(axis=0), train_tm.max(axis=0), mask)["RMSE"]
        print(f"epoch {epoch:02d}: train_mse={loss_sum / len(Xtr):.6f} val_normalized_rmse={val_score:.6f}")
        if val_score < best_score - 1e-7:
            best_score, best_epoch, bad = val_score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    model.load_state_dict(best_state)
    model.eval()
    pred_val = _predict(model, Xva, last_va, scale, n, device, args.batch)
    pred_test = _predict(model, Xte, last_te, scale, n, device, args.batch)

    def score(pred, truth):
        return {"original_units": metrics(pred, truth, mask),
                "per_pair_normalized": norm_metrics(pred, truth,
                                                     train_tm.min(axis=0),
                                                     train_tm.max(axis=0), mask)}

    naive_val = last_val.copy(); naive_test = last_test.copy()
    naive_val[:, np.arange(n), np.arange(n)] = 0.0
    naive_test[:, np.arange(n), np.arange(n)] = 0.0
    report = {
        "flags": vars(args),
        "selected_epoch": best_epoch,
        "best_validation_normalized_rmse": best_score,
        "validation": {"naive_last_value": score(naive_val, true_val), "flow_mlp": score(pred_val, true_val)},
        "test": {"naive_last_value": score(naive_test, true_test), "flow_mlp": score(pred_test, true_test)},
    }
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    np.save(f"{args.out}/pred_val.npy", pred_val)
    np.save(f"{args.out}/true_val.npy", true_val)
    np.save(f"{args.out}/last_observed_val.npy", last_val)
    np.save(f"{args.out}/pred_test.npy", pred_test)
    np.save(f"{args.out}/true_test.npy", true_test)
    np.save(f"{args.out}/last_observed_test.npy", last_test)
    with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved model and forecasts to {args.out}/")


if __name__ == "__main__":
    main()
