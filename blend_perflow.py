"""Validation-selected per-OD blend of two existing GEANT forecasters."""
import json
import os

import numpy as np

from data import load_tm, despike, metrics, norm_metrics


def main():
    # This focused experiment blends the two candidates using only validation
    # errors; test predictions are evaluated only after all weights are fixed.
    a, b, out = "res_PERFLOW_MEAN", "res_TREE_RANGE", "res_PERFLOW_BLEND"
    pa_v, pa_t = np.load(f"{a}/pred_val.npy"), np.load(f"{a}/pred_test.npy")
    pb_v, pb_t = np.load(f"{b}/pred_val.npy"), np.load(f"{b}/pred_test.npy")
    yv, yt = np.load(f"{a}/true_val.npy"), np.load(f"{a}/true_test.npy")
    assert np.array_equal(yv, np.load(f"{b}/true_val.npy"))
    assert np.array_equal(yt, np.load(f"{b}/true_test.npy"))
    nodes = yv.shape[-1]
    mask = ~np.eye(nodes, dtype=bool)
    tm, _ = despike(load_tm("geant_final.npy", nodes))
    lo, hi = tm[:int(len(tm) * .7)].min(axis=0), tm[:int(len(tm) * .7)].max(axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    weights = np.linspace(0, 1, 21)

    # Minimize validation squared error per OD pair (same as its normalized
    # RMSE because each pair uses one fixed training-range scale).
    err_by_pair = []
    for w in weights:
        pred = np.maximum((1 - w) * pa_v + w * pb_v, 0)
        err_by_pair.append((((pred - yv) / scale) ** 2).mean(axis=0))
    chosen = weights[np.argmin(np.stack(err_by_pair), axis=0)]
    chosen[~mask] = 0
    pv = np.maximum((1 - chosen) * pa_v + chosen * pb_v, 0)
    pt = np.maximum((1 - chosen) * pa_t + chosen * pb_t, 0)
    last_v = np.load(f"{a}/last_observed_val.npy")
    last_t = np.load(f"{a}/last_observed_test.npy")
    models_v = {"naive_last_value": last_v, "perflow_mean_ridge": pa_v,
                "flow_tree": pb_v, "perflow_validation_blend": pv}
    models_t = {"naive_last_value": last_t, "perflow_mean_ridge": pa_t,
                "flow_tree": pb_t, "perflow_validation_blend": pt}
    report = {
        "selection": "per-OD blending weights selected on validation normalized squared error; 21-point grid",
        "blend_weight_summary": {"mean": float(chosen[mask].mean()),
                                 "median": float(np.median(chosen[mask])),
                                 "fraction_tree_weight_positive": float((chosen[mask] > 0).mean())},
        "validation": {k: {"original_units": metrics(v, yv, mask),
                           "per_pair_normalized": norm_metrics(v, yv, lo, hi, mask)}
                       for k, v in models_v.items()},
        "test": {k: {"original_units": metrics(v, yt, mask),
                     "per_pair_normalized": norm_metrics(v, yt, lo, hi, mask)}
                 for k, v in models_t.items()}}
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/pred_val.npy", pv)
    np.save(f"{out}/true_val.npy", yv)
    np.save(f"{out}/pred_test.npy", pt)
    np.save(f"{out}/true_test.npy", yt)
    np.save(f"{out}/last_observed_val.npy", last_v)
    np.save(f"{out}/last_observed_test.npy", last_t)
    with open(f"{out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
