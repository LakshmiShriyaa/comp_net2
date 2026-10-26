"""Fair routing comparison: reactive vs proactive (vs hybrid, oracle) under the SAME rules.

Why this exists
    A proactive controller that acts at 45% utilisation will look better than a reactive one that acts
    at 70%, even with a useless forecast, simply because it acts earlier (and moves more flows).
    This script removes that confound:

      1. every scheme is replayed at every controller trigger in the same grid
      2. each scheme's trigger is then CHOSEN ON VALIDATION by one shared rule:
            lowest above-threshold traffic, subject to a reroute budget (flows moved per step)
      3. the chosen triggers are replayed on the test set
      4. differences (proactive minus reactive) come with 95% block-bootstrap intervals

Needs routing.py (the version with --link-metadata) in the same folder.

Inputs (all in the results folders; each is (steps, N, N) in kbps)
    --results      true_test.npy  pred_test.npy  last_observed_test.npy
    --val-results  true_val.npy   pred_val.npy   last_observed_val.npy      (optional, for trigger selection)

Example
    py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.35 0.45 0.55 0.70 --hybrid
"""
import argparse
import json

import numpy as np

from routing import (_capacity_vector, build_topology, load_link_metadata, PathTable,
                     optimize, optimize_overflow)

SCHEMES = {"reactive": "last", "proactive": "pred", "hybrid": "hyb", "oracle": "true"}


def load_split(folder, split):
    return tuple(np.load(f"{folder}/{name}_{split}.npy") for name in ("true", "pred", "last_observed"))


def replay(pt, cap, dem_true, dem_route, trigger, optimize_fn, report_thr):
    """Route each step with `dem_route` (None = static shortest paths), score on the TRUE demand."""
    zero = np.zeros(len(pt.flows), dtype=int)
    out = {k: [] for k in ("max_util", "above", "proxy", "physical", "rerouted")}
    for t in range(len(dem_true)):
        choice = zero if dem_route is None else optimize_fn(pt, dem_route[t], cap, trigger)
        load = pt.loads(dem_true[t], choice)
        util = load / cap
        total = max(float(load.sum()), 1e-9)
        out["max_util"].append(float(util.max()))
        out["above"].append(float(util.max() > report_thr))
        out["proxy"].append(float(np.maximum(load - report_thr * cap, 0).sum() / total))
        out["physical"].append(float(np.maximum(load - cap, 0).sum() / total))
        out["rerouted"].append(float((choice != 0).sum()))
    return {k: np.array(v) for k, v in out.items()}


def run_grid(pt, cap, split, triggers, optimize_fn, report_thr, schemes):
    true, pred, last = split
    dem = {"true": [pt.demand_vector(m) for m in true],
           "pred": [pt.demand_vector(m) for m in pred],
           "last": [pt.demand_vector(m) for m in last]}
    dem["hyb"] = [pt.demand_vector(np.maximum(p, l)) for p, l in zip(pred, last)]
    grid = {("static", None): replay(pt, cap, dem["true"], None, None, optimize_fn, report_thr)}
    for trig in triggers:
        for name in schemes:
            grid[(name, trig)] = replay(pt, cap, dem["true"], dem[SCHEMES[name]], trig, optimize_fn, report_thr)
            print(f"    done {name} @ {trig:.0%}", flush=True)
    return grid


def summary(r):
    return {"steps_above_%": 100 * r["above"].mean(), "above_traffic_%": 100 * r["proxy"].mean(),
            "mean_peak_util": r["max_util"].mean(), "physical_overload_%": 100 * r["physical"].mean(),
            "reroutes_per_step": r["rerouted"].mean()}


def block_ci(d, block=96, boot=2000, seed=0):
    """Mean of a per-step difference with a 95% moving-block bootstrap interval (blocks keep autocorrelation)."""
    d = np.asarray(d, dtype=float)
    T = len(d)
    block = min(block, T)
    nb = int(np.ceil(T / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T - block + 1, size=(boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(boot, -1)[:, :T]
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def choose_trigger(val_grid, name, triggers, budget):
    """Shared rule: least above-threshold traffic within the reroute budget; ties -> fewer reroutes."""
    rows = [(trig, summary(val_grid[(name, trig)])) for trig in triggers]
    ok = [(t, s) for t, s in rows if s["reroutes_per_step"] <= budget]
    if not ok:
        return min(rows, key=lambda r: r[1]["reroutes_per_step"])[0], False
    best = min(ok, key=lambda r: (round(r[1]["above_traffic_%"], 6), r[1]["steps_above_%"], r[1]["reroutes_per_step"]))
    return best[0], True


def print_grid(title, grid, triggers, schemes):
    print(f"\n{title}")
    head = f"{'trigger':<9}{'scheme':<11}{'steps>thr %':>12}{'traffic>thr %':>15}{'mean peak':>11}{'overload %':>12}{'reroutes':>10}"
    print(head)
    s = summary(grid[("static", None)])
    print(f"{'-':<9}{'static':<11}{s['steps_above_%']:>12.3f}{s['above_traffic_%']:>15.4f}{s['mean_peak_util']:>11.3f}"
          f"{s['physical_overload_%']:>12.4f}{s['reroutes_per_step']:>10.2f}")
    for trig in triggers:
        for name in schemes:
            s = summary(grid[(name, trig)])
            print(f"{trig:<9.0%}{name:<11}{s['steps_above_%']:>12.3f}{s['above_traffic_%']:>15.4f}{s['mean_peak_util']:>11.3f}"
                  f"{s['physical_overload_%']:>12.4f}{s['reroutes_per_step']:>10.2f}")


def print_diffs(grid, trig_pairs, boot, block):
    """proactive minus reactive, per step, for the given (reactive_trigger, proactive_trigger) pairs."""
    print(f"\nproactive MINUS reactive (negative = proactive better), mean [95% block-bootstrap CI], block={block} steps")
    print(f"{'reactive@':<11}{'proactive@':<12}{'steps>thr (pp)':>26}{'traffic>thr (pp)':>28}{'reroutes':>26}")
    rows = []
    for tr, tp in trig_pairs:
        a, b = grid[("proactive", tp)], grid[("reactive", tr)]
        cells = []
        for key, scale in (("above", 100), ("proxy", 100), ("rerouted", 1)):
            m, lo, hi = block_ci(scale * (a[key] - b[key]), block, boot)
            cells.append((m, lo, hi))
        rows.append({"reactive_trigger": tr, "proactive_trigger": tp,
                     "d_steps_above_pp": cells[0], "d_above_traffic_pp": cells[1], "d_reroutes": cells[2]})
        f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
        print(f"{tr:<11.0%}{tp:<12.0%}{f(cells[0]):>26}{f(cells[1]):>28}{f(cells[2]):>26}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder with true/pred/last_observed _test.npy")
    ap.add_argument("--val-results", default=None, help="folder with true/pred/last_observed _val.npy (trigger selection)")
    ap.add_argument("--nodes", type=int, default=23)
    ap.add_argument("--topo", default="geant")
    ap.add_argument("--edges", default=None)
    ap.add_argument("--link-metadata", required=True)
    ap.add_argument("--cap-scale", type=float, default=1.0,
                    help="multiply every link capacity by this fixed stress-test factor (e.g. 0.5); default 1.0")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--optimizer", choices=("minmax", "overflow"), default="overflow")
    ap.add_argument("--report-thr", type=float, default=0.70, help="utilisation level the 'above' metrics count")
    ap.add_argument("--triggers", type=float, nargs="+", default=[0.45, 0.55, 0.70],
                    help="controller trigger grid used by EVERY scheme")
    ap.add_argument("--budget", type=float, default=3.0, help="max average flows rerouted per step allowed when selecting a trigger")
    ap.add_argument("--hybrid", action="store_true", help="also replay the hybrid (max of forecast and last matrix)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=96, help="bootstrap block length in steps (96 = one day)")
    ap.add_argument("--out", default=None, help="where to write fair_routing_comparison.json (default: --results)")
    args = ap.parse_args()
    if args.cap_scale <= 0:
        ap.error("--cap-scale must be greater than zero")

    g = build_topology(args.topo, args.nodes, args.edges)
    load_link_metadata(g, args.link_metadata)
    pt = PathTable(g, args.k)
    cap = _capacity_vector(np.array([g[u][v]["capacity_kbps"] for u, v in pt.edges]), pt.E)
    cap *= args.cap_scale
    optimize_fn = optimize if args.optimizer == "minmax" else optimize_overflow
    schemes = ["reactive", "proactive", "oracle"] + (["hybrid"] if args.hybrid else [])
    triggers = sorted(args.triggers)
    print(f"{g.number_of_nodes()} nodes, {g.number_of_edges()} links, k={args.k}, optimizer={args.optimizer}, "
          f"capacity scale={args.cap_scale:g}; "
          f"'above' metrics count utilisation > {args.report_thr:.0%}; trigger grid {[f'{t:.0%}' for t in triggers]}")

    report = {"config": vars(args)}
    selected = {}
    selection_feasible = {}
    if args.val_results:
        val = load_split(args.val_results, "val")
        print(f"\nVALIDATION replay ({len(val[0])} steps)")
        vgrid = run_grid(pt, cap, val, triggers, optimize_fn, args.report_thr, schemes)
        print_grid("Validation grid (each scheme at every trigger)", vgrid, triggers, schemes)
        for name in schemes:
            if name == "oracle":
                continue
            trig, feasible = choose_trigger(vgrid, name, triggers, args.budget)
            selected[name] = trig
            selection_feasible[name] = feasible
            print(f"  selected on validation: {name:<9} -> trigger {trig:.0%}"
                  + ("" if feasible else f"   (no trigger met the {args.budget:g} reroutes/step budget; took the fewest reroutes)"))
        report["validation_selected_triggers"] = selected
        report["validation_reroute_budget_met"] = selection_feasible
        report["validation_grid"] = {f"{n}@{t}": summary(r) for (n, t), r in vgrid.items()}
        report["validation_note"] = ("If pred_val comes from a model fitted on validation (e.g. a stack), the forecast is "
                                     "in-sample there and validation selection favours it.")

    test = load_split(args.results, "test")
    print(f"\nTEST replay ({len(test[0])} steps)")
    tgrid = run_grid(pt, cap, test, triggers, optimize_fn, args.report_thr, schemes)
    print_grid("Test grid (each scheme at every trigger)", tgrid, triggers, schemes)
    report["test_grid"] = {f"{n}@{t}": summary(r) for (n, t), r in tgrid.items()}

    pairs = [(t, t) for t in triggers]                       # same trigger for both
    print("\n=== A. SAME trigger for reactive and proactive ===")
    report["same_trigger_diffs"] = print_diffs(tgrid, pairs, args.boot, args.block)
    if selected.get("reactive") is not None and selected.get("proactive") is not None:
        print("\n=== B. Each scheme at the trigger chosen on validation (same rule, same reroute budget) ===")
        sr, sp = selected["reactive"], selected["proactive"]
        for name, trig in (("reactive", sr), ("proactive", sp)):
            s = summary(tgrid[(name, trig)])
            print(f"  {name:<9}@{trig:.0%}: steps>thr {s['steps_above_%']:.3f}%  traffic>thr {s['above_traffic_%']:.4f}%  "
                  f"mean peak {s['mean_peak_util']:.3f}  reroutes {s['reroutes_per_step']:.2f}")
        report["validation_selected_diffs"] = print_diffs(tgrid, [(sr, sp)], args.boot, args.block)
    print("\nRead it like this: if the interval for 'steps>thr' and 'traffic>thr' includes 0, the forecast made no "
          "demonstrated difference to congestion; the reroutes column shows the cost of acting earlier.")

    import os
    out = args.out or args.results
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/fair_routing_comparison.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved {out}/fair_routing_comparison.json")


if __name__ == "__main__":
    main()
