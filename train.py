"""Train and evaluate CNN-LSTM against baselines.

Examples
    python train.py                                           # synthetic data, quick test
    python train.py --data geant_final.npy --nodes 23 --residual --mask-diag --horizon 4 --times geant_times.npy
    python train.py --data geant_final.npy --nodes 23 --mask-diag --shared --horizon 4 --times geant_times.npy --batch 64

Options (each is one experiment; change one at a time and use a different --out)
    --horizon H  forecast H steps ahead (H=4 is 1 hour for 15-minute GEANT data)
    --times F    timeline file from make_times.py: drops windows that straddle missing time steps
    --season S   steps in one day for the seasonal-naive baseline (96 for GEANT, 288 for Abilene; 0 = off)
    --shared     weight-shared per-OD-pair models (SharedCNNLSTM and SharedLSTM). They already
                 predict the change from the last matrix, so do not combine with --residual or --log
    --local-scale with --shared: normalise each pair by its own level in the input window instead of its
                 training-period average (handles traffic levels that drift over time)
    --residual   dense models only: predict the change from the last matrix (starts at naive)
    --mask-diag  zero the node-to-itself entries and exclude them from the metrics
    --log        log1p transform before scaling (dense models only; made results worse)
    --no-despike do not auto-repair grossly corrupt spike steps
    --under-w W  training loss weights under-prediction W times more than over-prediction (default 1 = plain MSE)
    --keras-style print epochs and test results in the Keras layout (Epoch k/N, loss, mae, val_loss, val_mae, Test MAE/RMSE)
    --tod        with --shared: time of day (sin/cos) as extra LSTM input
    --final      also retrain on train+val for the validation-chosen epoch count (extra '... + val retrain' row; pred_test.npy then holds that model)
    --skip-plain skip the second baseline model (plain LSTM, or shared LSTM without CNN)

Note: val_mse is in SCALED units and changes meaning with --log, so only compare the
final tables (original traffic units) across runs, and compare each model with the
Naive row of the SAME run.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import (load_tm, synthetic_tm, prepare, despike, metrics, rmse_by_step,
                  baseline_forecasts, norm_metrics, tod_features)
from model import CNNLSTM, PlainLSTM, Residual, SharedCNNLSTM

EVAL_CHUNK = 128      # windows per forward pass when evaluating (keeps memory small)


def fit(model, train, val, epochs, lr, batch, device, patience=8, under_w=1.0, style=False, nw=None,
        norm_loss=False):
    """train / val are tuples (X, Y) or (X, Y, TOD). With val=None the model trains for exactly
    `epochs` epochs (no early stopping). Returns (model, best_epoch).
    nw: optional (N, N) weights that turn scaled errors into per-pair-normalised errors. They are used
    for --keras-style display and, with norm_loss=True, for the training and early-stopping objective."""
    nwt = None if nw is None else torch.as_tensor(nw, dtype=torch.float32, device=device)

    def disp(p, y):
        e = p - y
        if nwt is not None:
            e = e * nwt
        return (e ** 2).sum().item(), e.abs().sum().item()

    fmt = (lambda v: f"{v:.4f}") if nwt is not None else (lambda v: f"{v:.3e}")
    tr = [torch.from_numpy(a) for a in train]
    dl = DataLoader(TensorDataset(*tr), batch_size=batch, shuffle=True)
    if val is not None:
        Xva, Yva, *Tva = [torch.from_numpy(a) for a in val]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, best_ep, bad = float("inf"), None, epochs, 0
    model.to(device)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tsq = tab = 0.0
        tcnt = 0
        for bt in dl:
            xb, yb, *ex = [t.to(device) for t in bt]
            opt.zero_grad()
            p = model(xb, *ex)
            w = torch.where(p < yb, under_w, 1.0)      # under-prediction costs under_w times more
            err = p - yb
            if norm_loss:
                if nwt is None:
                    raise ValueError("--norm-loss requires per-OD normalization weights")
                err = err * nwt
            loss = (w * err ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                q, r_ = disp(p, yb)
                tsq += q
                tab += r_
                tcnt += yb.numel()
        if val is None:
            nb, dt = len(dl), time.time() - t0
            if style:
                print(f"Epoch {ep}/{epochs}")
                print(f"{nb}/{nb} [{'=' * 20}] - {dt:.0f}s {1000 * dt / nb:.0f}ms/step - loss: {fmt(tsq / tcnt)} - mae: {tab / tcnt:.4f}")
            else:
                print(f"  epoch {ep:3d}  train_mse={tsq / tcnt:.3e}  ({dt:.0f}s)")
            continue
        model.eval()
        with torch.no_grad():
            sq = dsq = vab = 0.0
            for i in range(0, len(Xva), EVAL_CHUNK):
                pb = model(Xva[i:i + EVAL_CHUNK].to(device), *[t[i:i + EVAL_CHUNK].to(device) for t in Tva])
                yv = Yva[i:i + EVAL_CHUNK].to(device)
                err = pb - yv
                if norm_loss:
                    err = err * nwt
                sq += (err ** 2).sum().item()          # selected training objective: used for early stopping
                q, r_ = disp(pb, yv)
                dsq += q
                vab += r_
            vl = sq / Yva.numel()
        if style:
            nb, dt = len(dl), time.time() - t0
            print(f"Epoch {ep}/{epochs}")
            print(f"{nb}/{nb} [{'=' * 20}] - {dt:.0f}s {1000 * dt / nb:.0f}ms/step - loss: {fmt(tsq / tcnt)} - mae: "
                  f"{tab / tcnt:.4f} - val_loss: {fmt(dsq / Yva.numel())} - val_mae: {vab / Yva.numel():.4f}")
        else:
            print(f"  epoch {ep:3d}  val_mse={vl:.3e}  ({time.time() - t0:.0f}s)")
        if vl < best - 1e-9:
            best, bad, best_ep = vl, 0, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("  early stop")
                break
    if val is not None:
        model.load_state_dict(best_state)
    return model, best_ep


@torch.no_grad()
def predict(model, X, device, T=None):
    model.eval().to(device)
    outs = []
    for i in range(0, len(X), EVAL_CHUNK):
        ex = [torch.from_numpy(T[i:i + EVAL_CHUNK]).to(device)] if T is not None else []
        outs.append(model(torch.from_numpy(X[i:i + EVAL_CHUNK]).to(device), *ex).cpu().numpy())
    return np.concatenate(outs)


def build(kind, n, horizon, residual, shared, scale, local_scale=False, use_tod=False):
    if shared:
        return SharedCNNLSTM(n, horizon, scale, use_cnn=(kind == "cnn"), local_scale=local_scale, use_tod=use_tod)
    m = CNNLSTM(n, horizon=horizon) if kind == "cnn" else PlainLSTM(n, horizon=horizon)
    return Residual(m) if residual else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help=".npy/.csv traffic matrices; omit for synthetic data")
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--season", type=int, default=96)
    ap.add_argument("--times", default=None,
                    help="timeline .npy from make_times.py (minutes per step); drops windows that straddle gaps")
    ap.add_argument("--step-min", type=int, default=15, help="minutes between steps (15 GEANT, 5 Abilene)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=1,
                    help="CPU torch threads; a small value avoids oversubscription on shared machines")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keras-style", action="store_true",
                    help="print epochs as Epoch k/N + loss/mae/val_loss/val_mae and a Test MAE/RMSE/Prediction shape block (loss, mae in scaled units; test block in kbps)")
    ap.add_argument("--tod", action="store_true",
                    help="with --shared: also feed the time of day (sin/cos at every input step) to the LSTM")
    ap.add_argument("--final", action="store_true",
                    help="after choosing the epoch count on validation, retrain the CNN-LSTM from scratch on train+val "
                         "for that many epochs and report it as an extra row ('... + val retrain'); test data stays untouched")
    ap.add_argument("--under-w", type=float, default=1.0,
                    help="loss weight on under-prediction (>1 keeps peaks higher; try 2-4)")
    ap.add_argument("--norm-loss", action="store_true",
                    help="train and early-stop using per-OD range-normalized squared error (aligns with normalized RMSE metric)")
    ap.add_argument("--init-from", default=None,
                    help="optional CNN-LSTM state_dict checkpoint to fine-tune from")
    ap.add_argument("--out", default="results")
    ap.add_argument("--shared", action="store_true")
    ap.add_argument("--local-scale", action="store_true")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--mask-diag", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--no-despike", action="store_true")
    ap.add_argument("--skip-plain", action="store_true")
    ap.add_argument("--only-plain", action="store_true",
                    help="train and save only the plain/shared LSTM (useful for isolated model comparisons)")
    ap.add_argument("--train-stride", type=int, default=1,
                    help="keep every k-th training window to reduce CPU training time; validation stays complete")
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))
    if args.local_scale and not args.shared:
        raise SystemExit("--local-scale only applies together with --shared")
    if args.tod and not args.shared:
        raise SystemExit("--tod only applies together with --shared")
    if args.shared and args.log:
        raise SystemExit("--shared works in linear units; do not combine it with --log")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, H = args.nodes, args.horizon
    print(f"device={device}  horizon={H}  shared={args.shared}  residual={args.residual}  "
          f"mask_diag={args.mask_diag}  log={args.log}")

    if args.data:
        tm = load_tm(args.data, n)
        print(f"Loaded {args.data}: {tm.shape}")
    else:
        tm = synthetic_tm(T=4000, n_nodes=n, seed=args.seed)
        print(f"No --data given: using SYNTHETIC traffic {tm.shape} (pipeline test only)")

    if not args.no_despike:
        tm, fixed = despike(tm)
        print(f"despike: repaired {len(fixed)} corrupt step(s): {fixed[:20]}")

    times = np.load(args.times) if args.times else None
    if times is None:
        print("WARNING: no --times given, so missing time steps in the data are ignored (run make_times.py)")
    d = prepare(tm, window=args.window, horizon=H, log=args.log, mask_diag=args.mask_diag,
                times=times, step_min=args.step_min, season=args.season)
    print("windows kept / total:", {k: f"{a}/{b}" for k, (a, b) in d["stats"].items()})
    tod = tod_features(d, args.window, times, args.step_min) if args.tod else None
    if args.train_stride < 1:
        raise SystemExit("--train-stride must be >= 1")
    tr_set = tuple(x[::args.train_stride] for x in d["train"])
    if tod:
        tr_set += (tod["train"][::args.train_stride],)
    va_set = d["val"] + ((tod["val"],) if tod else ())
    te_tod = tod["test"] if tod else None
    sc = d["scaler"]
    Xva, Yva = d["val"]
    Xte, Yte = d["test"]
    true = sc.inverse(Yte)                       # (S, H, N, N) original units
    mask = ~np.eye(n, dtype=bool) if args.mask_diag else None

    scale = None
    if args.shared:
        level = sc.transform(d["tm"][:d["t_end"]]).mean(axis=0)            # typical scaled level per OD pair
        scale = np.maximum(level, 1e-3 * level.max()).astype(np.float32)

    tr_raw = d["tm"][:d["t_end"]]
    lo, hi = tr_raw.min(axis=0), tr_raw.max(axis=0)       # per-pair training range (for the normalised metrics)
    norm = {}

    nw = None                      # scaled error -> per-pair-normalised error (display only, linear scaling only)
    if not args.log:
        rng = hi - lo
        nw = np.where(rng > 0, (sc.hi - sc.lo + 1e-8) / np.where(rng > 0, rng, 1.0), 0.0).astype(np.float32)

    ref = float(((Xva[:, -1:] - Yva) ** 2).mean())
    print(f"reference: NAIVE val_mse in this run's scaled units = {ref:.3e}  (a model must get below this)")
    if nw is not None and args.keras_style:
        ref_n = float((((Xva[:, -1:] - Yva) * nw) ** 2).mean())
        ref_nm = float(np.abs((Xva[:, -1:] - Yva) * nw).mean())
        print(f"reference: NAIVE in per-pair normalised units (as in the Colab notebook): val_loss {ref_n:.4f}, val_mae {ref_nm:.4f}")
        print("--keras-style epoch lines below are in those per-pair normalised units; early stopping still uses the plain scaled loss")

    results, steps = {}, {}
    for name, p in baseline_forecasts(d, args.window, H, args.season).items():
        results[name] = metrics(p, true, mask)
        norm[name] = norm_metrics(p, true, lo, hi)
        steps[name] = rmse_by_step(p, true, mask)

    tag = " + ToD" if args.tod else ""

    def run(kind, label):
        def make():
            return build(kind, n, H, args.residual, args.shared, scale, args.local_scale, args.tod)

        def make_initialized():
            model = make()
            if args.init_from and kind == "cnn":
                model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=True))
            return model

        def score(model, name):
            pred = np.maximum(sc.inverse(predict(model, Xte, device, te_tod)), 0.0)     # (S, H, N, N)
            results[name] = metrics(pred, true, mask)
            norm[name] = norm_metrics(pred, true, lo, hi)
            steps[name] = rmse_by_step(pred, true, mask)
            return pred

        print(f"Training {label}")
        m, best_ep = fit(make_initialized(), tr_set, va_set, args.epochs, args.lr, args.batch, device,
                         args.patience, args.under_w, args.keras_style, nw, args.norm_loss)
        # Capture validation forecasts from the model selected without any
        # validation retraining; these are safe for validation-based blending.
        pred_val = np.maximum(sc.inverse(predict(m, Xva, device,
                                                tod["val"] if tod else None)), 0.0)
        pred = score(m, label)
        if args.final and kind == "cnn":
            print(f"Retraining {label} on train+val for {best_ep} epochs (no validation left, so the epoch count is fixed)")
            torch.manual_seed(args.seed)
            both = tuple(np.concatenate([x, y]) for x, y in zip(tr_set, va_set))
            m2, _ = fit(make_initialized(), both, None, best_ep, args.lr, args.batch, device, args.patience,
                        args.under_w, args.keras_style, nw, args.norm_loss)
            return m2, score(m2, label + " + val retrain"), pred, pred_val
        return m, pred, None, pred_val

    if args.only_plain:
        forecast_model, forecast_pred, valsel_pred, val_pred = run(
            "plain", ("Shared LSTM (no CNN)" if args.shared else "Plain LSTM") + tag)
        forecast_name = "plain_lstm.pt"
    else:
        if not args.skip_plain:
            run("plain", ("Shared LSTM (no CNN)" if args.shared else "Plain LSTM") + tag)
        forecast_model, forecast_pred, valsel_pred, val_pred = run(
            "cnn", ("Shared CNN-LSTM" if args.shared else "CNN-LSTM") + tag)
        forecast_name = "cnn_lstm.pt"

    # Keep the exact observation available when each test forecast was issued.

    print(f"\nAverage over all {H} forecast step(s) ahead")
    print("{:<28}{:>12}{:>12}{:>10}".format("Model", "MAE", "RMSE", "MAPE %"))
    for k, v in results.items():
        print("{:<28}{:>12.3f}{:>12.3f}{:>10.2f}".format(k, v["MAE"], v["RMSE"], v["MAPE_%"]))
    if H > 1:
        print("\nRMSE by steps ahead")
        print("{:<28}".format("Model") + "".join(f"{'+' + str(j + 1):>10}" for j in range(H)))
        for k, v in steps.items():
            print("{:<28}".format(k) + "".join(f"{x:>10.0f}" for x in v))
    if mask is not None:
        print("(scored on off-diagonal entries only)")
    print("\nNormalised metrics: every OD pair scaled by its own training range (per-pair min-max over all N*N entries, as in the Colab notebook)")
    print("{:<28}{:>12}{:>12}".format("Model", "MAE", "RMSE"))
    for k, v in norm.items():
        print("{:<28}{:>12.4f}{:>12.4f}".format(k, v["MAE"], v["RMSE"]))
    if args.keras_style:
        print("\nTest results (original units, kbps" + (", off-diagonal" if mask is not None else "") + ")")
        for k, v in results.items():
            print(f"{k}\n  Test MAE: {v['MAE']}\n  Test RMSE: {v['RMSE']}\n  Prediction shape: {tuple(true.shape)}")

    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/pred_test.npy", forecast_pred[:, -1])     # farthest step, for routing.py
    np.save(f"{args.out}/true_test.npy", true[:, -1])
    if valsel_pred is not None:
        np.save(f"{args.out}/pred_test_valsel.npy", valsel_pred[:, -1])   # the model chosen on validation only
    np.save(f"{args.out}/pred_test_all.npy", forecast_pred)
    np.save(f"{args.out}/true_test_all.npy", true)
    np.save(f"{args.out}/pred_val_all.npy", val_pred)
    np.save(f"{args.out}/true_val_all.npy", sc.inverse(Yva))
    np.save(f"{args.out}/last_observed_test.npy", sc.inverse(Xte[:, -1]))
    np.save(f"{args.out}/last_observed_val.npy", sc.inverse(Xva[:, -1]))
    np.save(f"{args.out}/history_test.npy", sc.inverse(Xte))
    np.save(f"{args.out}/history_val.npy", sc.inverse(Xva))
    np.save(f"{args.out}/test_window_start.npy", d["test_s"])
    torch.save(forecast_model.state_dict(), f"{args.out}/{forecast_name}")
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump({"flags": vars(args), "results": results, "normalised": norm, "rmse_by_step": steps}, f, indent=2)
    print(f"\nSaved predictions, model and metrics to {args.out}/")


if __name__ == "__main__":
    main()
