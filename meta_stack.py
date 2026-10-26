"""Chronological validation stacking of existing forecasts, then one test score."""
import json
import os
import numpy as np
from data import load_tm, despike, metrics, norm_metrics


def fit_ridge_stack(x, y, alpha):
    # x: samples × features × OD; y: samples × OD. Fit centered models per OD.
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm[None], y - ym[None]
    f, p = x.shape[1], x.shape[2]
    beta = np.empty((p, f), dtype=np.float64)
    intercept = np.empty(p, dtype=np.float64)
    eye = np.eye(f)
    for q in range(p):
        beta[q] = np.linalg.solve(xc[:, :, q].T @ xc[:, :, q] + alpha * eye,
                                  xc[:, :, q].T @ yc[:, q])
        intercept[q] = ym[q] - xm[:, q] @ beta[q]
    return beta, intercept


def apply_stack(x, beta, intercept):
    return np.einsum("sfq,qf->sq", x, beta) + intercept[None]


def fit_global_stack(x, y, alpha):
    # Shared model weights across OD pairs, with a learned per-flow intercept.
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm[None], y - ym[None]
    gram = np.einsum("sfq,srq->fr", xc, xc)
    rhs = np.einsum("sfq,sq->f", xc, yc)
    beta = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), rhs)
    intercept = ym - np.einsum("fq,f->q", xm, beta)
    return beta, intercept


def apply_global_stack(x, beta, intercept):
    return np.einsum("sfq,f->sq", x, beta) + intercept[None]


def fit_simplex(x, y, combos):
    # x is already candidate residual-vs-persistence; each row is a weight combo.
    # Fit an independent convex combination for each OD flow using only prefix data.
    best = np.full(x.shape[-1], np.inf)
    weights = np.zeros((x.shape[1], x.shape[-1]))
    for w in combos:
        pred = np.einsum("f,sfq->sq", w, x)
        loss = ((y - pred) ** 2).mean(axis=0)
        use = loss < best
        best[use] = loss[use]
        weights[:, use] = np.asarray(w)[:, None]
    return weights


def score_error(pred_delta, y_delta):
    return float(np.sqrt(np.mean((pred_delta - y_delta) ** 2)))


def main():
    out = "res_STACK"
    n = 23
    # Validation predictions from base models fitted on the training segment.
    cnn_v = np.mean([np.load(f"res_S4L{suffix}/pred_val_all.npy")[:, -1]
                     for suffix in ("", "_s1", "_s2")], axis=0)
    ridge_v = np.load("res_PERFLOW_MEAN/pred_val.npy")
    seasonal_v = np.load("res_LINEAR_SEASONAL/pred_val.npy")
    tree_v = np.mean([np.load("res_TREE_RANGE/pred_val.npy"),
                      np.load("res_TREE_FULL/pred_val.npy")], axis=0)
    log_v = np.load("res_LOG_FLOW/pred_val.npy")
    norm_cnn_v = np.load("res_NORM_CNN/pred_val_all.npy")[:, -1]
    roll_cnn_v = np.load("res_ROLL_CNN/pred_val.npy")
    context_v = np.load("res_FLOW_CONTEXT/pred_val.npy")
    hist_v = np.load("res_S4L/history_val.npy")
    ma_v = hist_v[:, -3:].mean(axis=1)
    base_v = np.load("res_PERFLOW_MEAN/last_observed_val.npy")
    yv = np.load("res_PERFLOW_MEAN/true_val.npy")

    # Test predictions from base models fitted on train+validation.
    cnn_t = np.mean([np.load(f"res_S4L{suffix}/pred_test_all.npy")[:, -1]
                     for suffix in ("", "_s1", "_s2")], axis=0)
    ridge_t = np.load("res_PERFLOW_MEAN/pred_test.npy")
    seasonal_t = np.load("res_LINEAR_SEASONAL/pred_test.npy")
    tree_t = np.mean([np.load("res_TREE_RANGE/pred_test.npy"),
                      np.load("res_TREE_FULL/pred_test.npy")], axis=0)
    log_t = np.load("res_LOG_FLOW/pred_test.npy")
    norm_cnn_t = np.load("res_NORM_CNN/pred_test_all.npy")[:, -1]
    roll_cnn_t = np.load("res_ROLL_CNN/pred_test.npy")
    context_t = np.load("res_FLOW_CONTEXT/pred_test.npy")
    hist_t = np.load("res_S4L/history_test.npy")
    ma_t = hist_t[:, -3:].mean(axis=1)
    base_t = np.load("res_PERFLOW_MEAN/last_observed_test.npy")
    yt = np.load("res_PERFLOW_MEAN/true_test.npy")
    assert np.array_equal(yv, np.load("res_TREE_RANGE/true_val.npy"))
    assert np.array_equal(yt, np.load("res_TREE_RANGE/true_test.npy"))
    mask = ~np.eye(n, dtype=bool)
    tm, _ = despike(load_tm("geant_final.npy", n))
    tr = tm[:int(len(tm) * .7)]
    lo, hi = tr.min(axis=0), tr.max(axis=0)
    scale = np.maximum(hi - lo, 1.0).reshape(-1)

    # Flatten only off-diagonal pairs. Each candidate is expressed as a change
    # from persistence, so the learner estimates corrections to the baseline.
    names = ["cnn_seed_ensemble", "normalized_loss_cnn", "perflow_mean_ridge",
             "seasonal_ridge", "log_flow", "tree_ensemble", "moving_average_3",
             "recursive_cnn", "flow_context"]
    pv = [cnn_v, norm_cnn_v, ridge_v, seasonal_v, log_v, tree_v, ma_v, roll_cnn_v, context_v]
    pt = [cnn_t, norm_cnn_t, ridge_t, seasonal_t, log_t, tree_t, ma_t, roll_cnn_t, context_t]
    xv = np.stack([(p - base_v)[:, mask] / scale[mask.ravel()] for p in pv], axis=1)
    xt = np.stack([(p - base_t)[:, mask] / scale[mask.ravel()] for p in pt], axis=1)
    y_delta = ((yv - base_v)[:, mask] / scale[mask.ravel()]).astype(np.float64)
    y_test_delta = ((yt - base_t)[:, mask] / scale[mask.ravel()]).astype(np.float64)

    # Expanding-window validation folds: fit meta-weights on earlier validation
    # forecasts, score on subsequent unseen validation blocks.
    folds = [(int(.50 * len(yv)), int(.75 * len(yv))),
             (int(.75 * len(yv)), len(yv))]
    alphas = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    cap_extras = (None, 0.0, 0.5, 1.0, 2.0)
    stack_scores = {(a, cap): [] for a in alphas for cap in cap_extras}
    global_scores = {(a, cap): [] for a in alphas for cap in cap_extras}
    baseline_scores = []
    # Coefficients may sum to less than one; the remainder stays on persistence.
    simplex = [(a / 5, b / 5, c / 5, d / 5, e / 5, f / 5, g / 5, h / 5, z / 5)
               for a in range(6) for b in range(6-a) for c in range(6-a-b)
               for d in range(6-a-b-c) for e in range(6-a-b-c-d)
               for f in range(6-a-b-c-d-e) for g in range(6-a-b-c-d-e-f)
               for h in range(6-a-b-c-d-e-f-g) for z in range(6-a-b-c-d-e-f-g-h+1)]
    simple_scores = []
    for start, end in folds:
        baseline_scores.append(score_error(np.zeros_like(y_delta[start:end]), y_delta[start:end]))
        for alpha in alphas:
            beta, bias = fit_ridge_stack(xv[:start], y_delta[:start], alpha)
            raw_delta = apply_stack(xv[start:end], beta, bias)
            raw_level = base_v[start:end][:, mask] + raw_delta * scale[mask.ravel()]
            cap = hi[mask].reshape(1, -1)
            rng = scale[mask.ravel()].reshape(1, -1)
            for extra in cap_extras:
                clipped = raw_level if extra is None else np.minimum(raw_level, cap + extra * rng)
                clipped = np.maximum(clipped, 0.0)
                clipped_delta = (clipped - base_v[start:end][:, mask]) / rng
                stack_scores[(alpha, extra)].append(score_error(clipped_delta, y_delta[start:end]))
            beta_g, bias_g = fit_global_stack(xv[:start], y_delta[:start], alpha)
            raw_delta_g = apply_global_stack(xv[start:end], beta_g, bias_g)
            raw_level_g = base_v[start:end][:, mask] + raw_delta_g * scale[mask.ravel()]
            for extra in cap_extras:
                clipped = raw_level_g if extra is None else np.minimum(raw_level_g, cap + extra * rng)
                clipped = np.maximum(clipped, 0.0)
                clipped_delta = (clipped - base_v[start:end][:, mask]) / rng
                global_scores[(alpha, extra)].append(score_error(clipped_delta, y_delta[start:end]))
        weights = fit_simplex(xv[:start], y_delta[:start], simplex)
        pred = np.einsum("sfq,fq->sq", xv[start:end], weights)
        simple_scores.append(score_error(pred, y_delta[start:end]))
    alpha, cap_extra = min(stack_scores, key=lambda key: np.mean(stack_scores[key]))
    stack_cv = float(np.mean(stack_scores[(alpha, cap_extra)]))
    global_alpha, global_cap = min(global_scores, key=lambda key: np.mean(global_scores[key]))
    global_cv = float(np.mean(global_scores[(global_alpha, global_cap)]))
    simplex_cv = float(np.mean(simple_scores))
    baseline_cv = float(np.mean(baseline_scores))
    chosen = min(((stack_cv, "ridge_stack"), (global_cv, "global_stack"),
                  (simplex_cv, "convex_simplex")))[1]
    print(json.dumps({"validation_cv_normalized_rmse": {"selected_strategy": chosen,
          "selected_stack_alpha": alpha, "selected_upper_cap_extra_ranges": cap_extra,
          "ridge_stack": stack_cv, "global_stack_alpha": global_alpha,
          "global_stack_upper_cap_extra_ranges": global_cap, "global_stack": global_cv,
          "convex_simplex": simplex_cv, "persistence": baseline_cv,
          "ridge_stack_improvement_vs_persistence_%": 100 * (1 - stack_cv / baseline_cv),
          "simplex_grid_points": len(simplex),
          "folds": folds}}, indent=2))

    # Refit selected meta-model on all validation predictions, then evaluate test.
    if chosen == "ridge_stack":
        beta, bias = fit_ridge_stack(xv, y_delta, alpha)
        test_delta = apply_stack(xt, beta, bias)
        val_delta = apply_stack(xv, beta, bias)
        saved_weights = {"beta": beta.tolist(), "intercept": bias.tolist(),
                         "upper_cap_extra_ranges": cap_extra}
    elif chosen == "global_stack":
        alpha, cap_extra = global_alpha, global_cap
        beta, bias = fit_global_stack(xv, y_delta, alpha)
        test_delta = apply_global_stack(xt, beta, bias)
        val_delta = apply_global_stack(xv, beta, bias)
        saved_weights = {"global_beta": beta.tolist(), "per_od_intercept": bias.tolist(),
                         "upper_cap_extra_ranges": cap_extra}
    else:
        weights = fit_simplex(xv, y_delta, simplex)
        val_delta = np.einsum("sfq,fq->sq", xv, weights)
        test_delta = np.einsum("sfq,fq->sq", xt, weights)
        saved_weights = {"mean_candidate_weights": weights.mean(axis=1).tolist(),
                         "weights_by_od": weights.tolist()}
    pred_val, pred_test = base_v.copy(), base_t.copy()
    pred_val[:, mask] += val_delta * scale[mask.ravel()]
    pred_test[:, mask] += test_delta * scale[mask.ravel()]
    pred_val, pred_test = np.maximum(pred_val, 0), np.maximum(pred_test, 0)
    if chosen in ("ridge_stack", "global_stack") and cap_extra is not None:
        upper = hi[mask] + cap_extra * scale[mask.ravel()]
        pred_val[:, mask] = np.minimum(pred_val[:, mask], upper[None])
        pred_test[:, mask] = np.minimum(pred_test[:, mask], upper[None])
    for p in (pred_val, pred_test):
        p[:, np.arange(n), np.arange(n)] = 0
    report = {"selection": {"strategy": chosen, "ridge_alpha": alpha,
              "upper_cap_extra_ranges": cap_extra,
              "validation_cv_normalized_rmse": {"persistence": baseline_cv,
                    "ridge_stack": stack_cv, "convex_simplex": simplex_cv,
                    "ridge_stack_improvement_vs_persistence_%": 100 * (1 - stack_cv / baseline_cv)}},
              "model_features": names,
              "validation": {"persistence": {"original_units": metrics(base_v, yv, mask),
                              "normalized": norm_metrics(base_v, yv, lo, hi, mask)},
                             "stack": {"original_units": metrics(pred_val, yv, mask),
                                       "normalized": norm_metrics(pred_val, yv, lo, hi, mask)}},
              "test": {"persistence": {"original_units": metrics(base_t, yt, mask),
                          "normalized": norm_metrics(base_t, yt, lo, hi, mask)},
                       "stack": {"original_units": metrics(pred_test, yt, mask),
                                 "normalized": norm_metrics(pred_test, yt, lo, hi, mask)}}}
    os.makedirs(out, exist_ok=True)
    for name, p in (("pred_val", pred_val), ("true_val", yv), ("last_observed_val", base_v),
                    ("pred_test", pred_test), ("true_test", yt), ("last_observed_test", base_t)):
        np.save(f"{out}/{name}.npy", p)
    with open(f"{out}/stack_weights.json", "w", encoding="utf-8") as f:
        json.dump(saved_weights, f)
    with open(f"{out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
