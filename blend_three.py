"""Per-OD validation-selected blend of persistence, ridge, and tree forecasts."""
import json
import os
import numpy as np
from data import load_tm, despike, metrics, norm_metrics


def main():
    ridge, tree = "res_PERFLOW_MEAN", "res_TREE_RANGE"
    out = "res_TRIPLE_BLEND"
    rv, rt = np.load(f"{ridge}/pred_val.npy"), np.load(f"{ridge}/pred_test.npy")
    tv, tt = np.load(f"{tree}/pred_val.npy"), np.load(f"{tree}/pred_test.npy")
    yv, yt = np.load(f"{ridge}/true_val.npy"), np.load(f"{ridge}/true_test.npy")
    assert np.array_equal(yv, np.load(f"{tree}/true_val.npy"))
    assert np.array_equal(yt, np.load(f"{tree}/true_test.npy"))
    lv, lt = np.load(f"{ridge}/last_observed_val.npy"), np.load(f"{ridge}/last_observed_test.npy")
    n = yv.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    tm, _ = despike(load_tm("geant_final.npy", n))
    lo, hi = tm[:int(len(tm) * .7)].min(axis=0), tm[:int(len(tm) * .7)].max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)

    # Coarse simplex grid; choose independent weights for each OD pair solely
    # from validation normalized squared error.
    combos = [(i / 10, j / 10, (10 - i - j) / 10)
              for i in range(11) for j in range(11 - i)]
    losses = []
    for wr, wt, wl in combos:
        p = np.maximum(wr * rv + wt * tv + wl * lv, 0)
        losses.append((((p - yv) / scale) ** 2).mean(axis=0))
    best = np.argmin(np.stack(losses), axis=0)
    weights = np.asarray(combos)[best]
    weights[~mask] = (0, 0, 1)
    pv = np.maximum(weights[..., 0] * rv + weights[..., 1] * tv + weights[..., 2] * lv, 0)
    pt = np.maximum(weights[..., 0] * rt + weights[..., 1] * tt + weights[..., 2] * lt, 0)
    val_models = {"naive_last_value": lv, "perflow_mean_ridge": rv, "flow_tree": tv,
                  "perflow_triple_blend": pv}
    test_models = {"naive_last_value": lt, "perflow_mean_ridge": rt, "flow_tree": tt,
                   "perflow_triple_blend": pt}
    report = {
        "selection": "per-OD simplex weights selected on validation normalized squared error; 66-point 0.1 grid",
        "mean_weights_ridge_tree_persistence": weights[mask].mean(axis=0).tolist(),
        "validation": {k: {"original_units": metrics(p, yv, mask),
                           "per_pair_normalized": norm_metrics(p, yv, lo, hi, mask)} for k, p in val_models.items()},
        "test": {k: {"original_units": metrics(p, yt, mask),
                     "per_pair_normalized": norm_metrics(p, yt, lo, hi, mask)} for k, p in test_models.items()}}
    os.makedirs(out, exist_ok=True)
    for name, arr in (("pred_val", pv), ("true_val", yv), ("pred_test", pt),
                      ("true_test", yt), ("last_observed_val", lv), ("last_observed_test", lt)):
        np.save(f"{out}/{name}.npy", arr)
    np.save(f"{out}/weights_by_od.npy", weights)
    with open(f"{out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
