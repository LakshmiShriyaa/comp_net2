"""Validation-selected blend of a CNN-LSTM and the per-flow ridge forecast.

The ridge weight is chosen by validation MAE only. The test set is used only for
the final comparison, and both input result folders must contain identical target
windows.

Example:
    python blend_forecasts.py --cnn res_S4L --ridge res_AR --out res_FINAL \
        --data geant_final.npy --nodes 23 --mask-diag
"""
import argparse
import json
import os

import numpy as np

from data import load_tm, despike, metrics, norm_metrics


def _last_step(path):
    pred = np.load(path)
    return pred[:, -1] if pred.ndim == 4 else pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnn", required=True, help="CNN-LSTM training output folder")
    ap.add_argument("--ridge", required=True, help="ridge_forecast.py output folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--mask-diag", action="store_true")
    args = ap.parse_args()

    cnn_val = _last_step(f"{args.cnn}/pred_val_all.npy")
    cnn_test = _last_step(f"{args.cnn}/pred_test_all.npy")
    ridge_val = _last_step(f"{args.ridge}/pred_val.npy")
    ridge_test = _last_step(f"{args.ridge}/pred_test.npy")
    true_val = _last_step(f"{args.cnn}/true_val_all.npy")
    true_test = _last_step(f"{args.cnn}/true_test_all.npy")
    if not np.array_equal(true_val, np.load(f"{args.ridge}/true_val.npy")):
        raise SystemExit("CNN and ridge validation targets differ; refusing to blend")
    if not np.array_equal(true_test, np.load(f"{args.ridge}/true_test.npy")):
        raise SystemExit("CNN and ridge test targets differ; refusing to blend")

    mask = ~np.eye(args.nodes, dtype=bool) if args.mask_diag else None
    weights = np.linspace(0.0, 1.0, 21)
    val_candidates = [np.maximum((1 - w) * cnn_val + w * ridge_val, 0.0) for w in weights]
    val_errors = [metrics(p, true_val, mask)["MAE"] for p in val_candidates]
    best_idx = int(np.argmin(val_errors))
    weight = float(weights[best_idx])
    pred_val = val_candidates[best_idx]
    pred_test = np.maximum((1 - weight) * cnn_test + weight * ridge_test, 0.0)

    last_val = np.load(f"{args.cnn}/last_observed_val.npy")
    last_test = np.load(f"{args.cnn}/last_observed_test.npy")
    history_val_path = f"{args.cnn}/history_val.npy"
    history_test_path = f"{args.cnn}/history_test.npy"
    if os.path.exists(history_val_path) and os.path.exists(history_test_path):
        history_val = np.load(history_val_path)
        history_test = np.load(history_test_path)
    else:
        history_val = history_test = None
    if mask is not None:
        for a in (last_val, last_test):
            a[:, np.arange(args.nodes), np.arange(args.nodes)] = 0.0
    models_val = {"Naive (last value)": last_val,
                  "CNN-LSTM": cnn_val,
                  "Per-OD ridge": ridge_val,
                  "CNN-LSTM + ridge blend": pred_val}
    models_test = {"Naive (last value)": last_test,
                   "CNN-LSTM": cnn_test,
                   "Per-OD ridge": ridge_test,
                   "CNN-LSTM + ridge blend": pred_test}
    if history_val is not None:
        ma_val = history_val[:, -3:].mean(axis=1)
        ma_test = history_test[:, -3:].mean(axis=1)
        if mask is not None:
            for a in (ma_val, ma_test):
                a[:, np.arange(args.nodes), np.arange(args.nodes)] = 0.0
        models_val["Moving average (3)"] = ma_val
        models_test["Moving average (3)"] = ma_test

    tm, _ = despike(load_tm(args.data, args.nodes))
    if args.mask_diag:
        tm[:, np.arange(args.nodes), np.arange(args.nodes)] = 0.0
    train = tm[:int(len(tm) * 0.7)]
    report = {
        "flags": vars(args),
        "selection": {"criterion": "validation MAE",
                      "ridge_weight": weight,
                      "validation_candidate_weights": weights.tolist(),
                      "validation_mae_by_weight": val_errors},
        "validation": {name: {"original_units": metrics(p, true_val, mask),
                              "per_pair_normalized": norm_metrics(p, true_val,
                                                                   train.min(axis=0),
                                                                   train.max(axis=0), mask)}
                       for name, p in models_val.items()},
        "test": {name: {"original_units": metrics(p, true_test, mask),
                        "per_pair_normalized": norm_metrics(p, true_test,
                                                             train.min(axis=0),
                                                             train.max(axis=0), mask)}
                 for name, p in models_test.items()},
    }
    os.makedirs(args.out, exist_ok=True)
    np.save(f"{args.out}/pred_val.npy", pred_val)
    np.save(f"{args.out}/true_val.npy", true_val)
    np.save(f"{args.out}/pred_test.npy", pred_test)
    np.save(f"{args.out}/true_test.npy", true_test)
    np.save(f"{args.out}/last_observed_val.npy", last_val)
    np.save(f"{args.out}/last_observed_test.npy", last_test)
    if history_val is not None:
        np.save(f"{args.out}/history_val.npy", history_val)
        np.save(f"{args.out}/history_test.npy", history_test)
    with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"selected_ridge_weight": weight,
                      "validation_mae": report["validation"]["CNN-LSTM + ridge blend"]["original_units"]["MAE"],
                      "test": report["test"]}, indent=2))
    print(f"Saved validation-selected forecasts to {args.out}/")


if __name__ == "__main__":
    main()
