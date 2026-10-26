"""Validation-selected per-OD four-way ensemble for the GEANT forecasting task."""
import json
import os
import numpy as np
from data import load_tm, despike, metrics, norm_metrics


def last(path):
    p = np.load(path)
    return p[:, -1] if p.ndim == 4 else p


def main():
    # Candidate weights are fitted using validation only. The test set is
    # touched only after per-OD weights have been selected and frozen.
    cnn_v, cnn_t = last("res_S4L/pred_val_all.npy"), last("res_S4L/pred_test_all.npy")
    ridge_v, ridge_t = (np.load("res_PERFLOW_MEAN/pred_val.npy"),
                        np.load("res_PERFLOW_MEAN/pred_test.npy"))
    tree_v, tree_t = np.load("res_TREE_RANGE/pred_val.npy"), np.load("res_TREE_RANGE/pred_test.npy")
    base_v, base_t = (np.load("res_PERFLOW_MEAN/last_observed_val.npy"),
                      np.load("res_PERFLOW_MEAN/last_observed_test.npy"))
    yv, yt = last("res_S4L/true_val_all.npy"), last("res_S4L/true_test_all.npy")
    assert np.array_equal(yv, np.load("res_PERFLOW_MEAN/true_val.npy"))
    assert np.array_equal(yt, np.load("res_PERFLOW_MEAN/true_test.npy"))
    candidates_v = np.stack([cnn_v, ridge_v, tree_v, base_v])
    candidates_t = np.stack([cnn_t, ridge_t, tree_t, base_t])
    nodes = yv.shape[-1]
    mask = ~np.eye(nodes, dtype=bool)
    tm, _ = despike(load_tm("geant_final.npy", nodes))
    lo, hi = tm[:int(len(tm) * .7)].min(axis=0), tm[:int(len(tm) * .7)].max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)

    # Four-part simplex with 0.1 increments (286 possible combinations).
    combos = [(a / 10, b / 10, c / 10, (10 - a - b - c) / 10)
              for a in range(11) for b in range(11 - a)
              for c in range(11 - a - b)]
    best_loss = np.full((nodes, nodes), np.inf)
    best_w = np.zeros((nodes, nodes, 4), dtype=np.float32)
    # Iterate candidate combinations without allocating a huge combination×time tensor.
    for combo in combos:
        pred = np.maximum(np.einsum("m,mtnk->tnk", combo, candidates_v), 0)
        loss = (((pred - yv) / scale) ** 2).mean(axis=0)
        update = loss < best_loss
        best_loss[update] = loss[update]
        best_w[update] = combo
    best_w[~mask] = (0, 0, 0, 1)
    pv = np.maximum(np.einsum("mnk,mtnk->tnk", best_w.transpose(2, 0, 1), candidates_v), 0)
    pt = np.maximum(np.einsum("mnk,mtnk->tnk", best_w.transpose(2, 0, 1), candidates_t), 0)
    names = ["cnn_lstm", "perflow_mean_ridge", "flow_tree", "naive_last_value"]
    val_models = {name: candidates_v[i] for i, name in enumerate(names)}
    test_models = {name: candidates_t[i] for i, name in enumerate(names)}
    val_models["four_way_validation_blend"] = pv
    test_models["four_way_validation_blend"] = pt
    report = {
        "selection": "per-OD weights selected on validation normalized squared error; 286-point four-model simplex grid (0.1 increments)",
        "model_order": names,
        "mean_weights": best_w[mask].mean(axis=0).tolist(),
        "validation": {k: {"original_units": metrics(p, yv, mask),
                           "per_pair_normalized": norm_metrics(p, yv, lo, hi, mask)} for k, p in val_models.items()},
        "test": {k: {"original_units": metrics(p, yt, mask),
                     "per_pair_normalized": norm_metrics(p, yt, lo, hi, mask)} for k, p in test_models.items()}}
    out = "res_FOUR_BLEND"
    os.makedirs(out, exist_ok=True)
    for name, arr in (("pred_val", pv), ("true_val", yv), ("pred_test", pt),
                      ("true_test", yt), ("last_observed_val", base_v),
                      ("last_observed_test", base_t), ("weights_by_od", best_w)):
        np.save(f"{out}/{name}.npy", arr)
    with open(f"{out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
