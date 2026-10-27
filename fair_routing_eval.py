"""Fair routing comparison: reactive vs proactive (vs hybrid, oracle) under the SAME rules.

Why this exists
    A proactive controller that acts at 45% utilisation will look better than a reactive one that acts
    at 70%, even with a useless forecast, simply because it acts earlier (and moves more flows).
    This script removes that confound:

      1. every scheme is replayed at every controller trigger in the same grid
      2. each scheme's trigger is then CHOSEN ON VALIDATION by one shared rule:
            lowest above-threshold traffic, subject to a budget on average flows assigned to non-default paths
      3. the chosen triggers are replayed on the test set
      4. differences (proactive minus reactive) come with 95% block-bootstrap intervals

Persistence (--persistence): every scheme is also replayed with route hysteresis at the same triggers; the
hysteresis is chosen per scheme and trigger on validation by ONE rule (no more congestion than independent
routing, stretch <= 1.02, then fewest route changes). Section C checks 'persistence stays at/below the
independent level' pair by pair; section D compares persistent proactive with persistent reactive.

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


def optimize_stable(pt, demand, cap, trigger, previous, hysteresis, objective):
    """Persistence rule (same as routing_stability_sweep.py): start from the previous paths and move a flow
    only if the move cuts the predicted peak (minmax) or squared excess over trigger (overflow) by at least
    `hysteresis` x its current value."""
    choice = previous.copy()
    load = pt.loads(demand, choice)

    def overflow_cost(values):
        excess = np.maximum(values / cap - trigger, 0.0)
        return float(excess @ excess)

    current_cost = overflow_cost(load)
    for _ in range(100):
        util = load / cap
        worst = int(util.argmax())
        current_max = float(util[worst])
        if current_max <= trigger or (objective == "overflow" and current_cost <= 1e-12):
            break
        on_worst = np.where((pt.A[np.arange(len(demand)), choice, worst] > 0) & (demand > 0))[0]
        required = (max(1e-6, hysteresis * current_max) if objective == "minmax"
                    else max(1e-12, hysteresis * current_cost))
        best_score = current_max - required if objective == "minmax" else current_cost - required
        best_move = None
        for flow in on_worst:
            old = pt.A[flow, choice[flow]]
            for path in range(pt.k):
                if path == choice[flow] or not pt.valid[flow, path]:
                    continue
                new_load = load - demand[flow] * old + demand[flow] * pt.A[flow, path]
                score = (float((new_load / cap).max()) if objective == "minmax" else overflow_cost(new_load))
                if score < best_score:
                    best_score, best_move = score, (flow, path, new_load)
        if best_move is None:
            break
        flow, path, load = best_move
        choice[flow] = path
        current_cost = overflow_cost(load)
    return choice


def replay(pt, cap, dem_true, dem_route, trigger, optimize_fn, report_thr, hyst=None, objective="overflow"):
    """Route each step with `dem_route` (None = static shortest paths), score on the TRUE demand.
    hyst=None -> every interval is optimised independently; hyst=h -> persistent routes with hysteresis h."""
    zero = np.zeros(len(pt.flows), dtype=int)
    out = {k: [] for k in ("max_util", "above", "proxy", "physical", "rerouted", "changes", "stretch")}
    costs = getattr(pt, "costs", None)
    previous = zero
    for t in range(len(dem_true)):
        if dem_route is None:
            choice = zero
        elif hyst is None:
            choice = optimize_fn(pt, dem_route[t], cap, trigger)
        else:
            choice = optimize_stable(pt, dem_route[t], cap, trigger, previous, hyst, objective)
        load = pt.loads(dem_true[t], choice)
        util = load / cap
        total = max(float(load.sum()), 1e-9)
        out["max_util"].append(float(util.max()))
        out["above"].append(float(util.max() > report_thr))
        out["proxy"].append(float(np.maximum(load - report_thr * cap, 0).sum() / total))
        out["physical"].append(float(np.maximum(load - cap, 0).sum() / total))
        out["rerouted"].append(float((choice != 0).sum()))
        out["changes"].append(float((choice != previous).sum()))
        if costs is not None:
            idx = np.arange(len(choice))
            out["stretch"].append(float((dem_true[t] * costs[idx, choice]).sum()
                                        / max(float((dem_true[t] * costs[:, 0]).sum()), 1e-9)))
        else:
            out["stretch"].append(float("nan"))
        previous = choice
    return {k: np.array(v) for k, v in out.items()}


def run_grid(pt, cap, split, triggers, optimize_fn, report_thr, schemes, hysts=(), objective="overflow"):
    """Keys: ('static',None,None); (scheme, trigger, None) independent; (scheme, trigger, h) persistent."""
    true, pred, last = split
    dem = {"true": [pt.demand_vector(m) for m in true],
           "pred": [pt.demand_vector(m) for m in pred],
           "last": [pt.demand_vector(m) for m in last]}
    dem["hyb"] = [pt.demand_vector(np.maximum(p, l)) for p, l in zip(pred, last)]
    grid = {("static", None, None): replay(pt, cap, dem["true"], None, None, optimize_fn, report_thr)}
    for trig in triggers:
        for name in schemes:
            grid[(name, trig, None)] = replay(pt, cap, dem["true"], dem[SCHEMES[name]], trig, optimize_fn, report_thr)
            for h in hysts:
                grid[(name, trig, h)] = replay(pt, cap, dem["true"], dem[SCHEMES[name]], trig, optimize_fn,
                                               report_thr, hyst=h, objective=objective)
            print(f"    done {name} @ {trig:.0%}" + (f" (+{len(hysts)} persistence settings)" if hysts else ""), flush=True)
    return grid


def summary(r):
    return {"steps_above_%": 100 * r["above"].mean(), "above_traffic_%": 100 * r["proxy"].mean(),
            "mean_peak_util": r["max_util"].mean(), "physical_overload_%": 100 * r["physical"].mean(),
            "flows_off_default_per_step": r["rerouted"].mean(), "changes_per_step": r["changes"].mean(),
            "stretch": float(np.nanmean(r["stretch"]))}


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
    """Shared rule: least above-threshold traffic within the off-default-flow budget."""
    rows = [(trig, summary(val_grid[(name, trig, None)])) for trig in triggers]
    ok = [(t, s) for t, s in rows if s["flows_off_default_per_step"] <= budget]
    if not ok:
        return min(rows, key=lambda r: r[1]["flows_off_default_per_step"])[0], False
    best = min(ok, key=lambda r: (round(r[1]["above_traffic_%"], 6), r[1]["steps_above_%"], r[1]["flows_off_default_per_step"]))
    return best[0], True


def print_grid(title, grid, triggers, schemes):
    print(f"\n{title}")
    head = f"{'trigger':<9}{'scheme':<11}{'steps>thr %':>12}{'traffic>thr %':>15}{'mean peak':>11}{'overload %':>12}{'off-default':>12}"
    print(head)
    s = summary(grid[("static", None, None)])
    print(f"{'-':<9}{'static':<11}{s['steps_above_%']:>12.3f}{s['above_traffic_%']:>15.4f}{s['mean_peak_util']:>11.3f}"
          f"{s['physical_overload_%']:>12.4f}{s['flows_off_default_per_step']:>12.2f}")
    for trig in triggers:
        for name in schemes:
            s = summary(grid[(name, trig, None)])
            print(f"{trig:<9.0%}{name:<11}{s['steps_above_%']:>12.3f}{s['above_traffic_%']:>15.4f}{s['mean_peak_util']:>11.3f}"
                  f"{s['physical_overload_%']:>12.4f}{s['flows_off_default_per_step']:>12.2f}")


def print_diffs(grid, trig_pairs, boot, block):
    """proactive minus reactive, per step, for the given (reactive_trigger, proactive_trigger) pairs."""
    print(f"\nproactive MINUS reactive (negative = proactive better), mean [95% block-bootstrap CI], block={block} steps")
    print(f"{'reactive@':<11}{'proactive@':<12}{'steps>thr (pp)':>26}{'traffic>thr (pp)':>28}{'off-default flows':>26}")
    rows = []
    for tr, tp in trig_pairs:
        a, b = grid[("proactive", tp, None)], grid[("reactive", tr, None)]
        cells = []
        for key, scale in (("above", 100), ("proxy", 100), ("rerouted", 1)):
            m, lo, hi = block_ci(scale * (a[key] - b[key]), block, boot)
            cells.append((m, lo, hi))
        rows.append({"reactive_trigger": tr, "proactive_trigger": tp,
                     "d_steps_above_pp": cells[0], "d_above_traffic_pp": cells[1], "d_flows_off_default": cells[2]})
        f = lambda c: f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"
        print(f"{tr:<11.0%}{tp:<12.0%}{f(cells[0]):>26}{f(cells[1]):>28}{f(cells[2]):>26}")
    return rows


PERSIST_METRICS = (("above", 100, "steps>thr (pp)"), ("proxy", 100, "traffic>thr (pp)"),
                   ("max_util", 1, "mean peak util"), ("changes", 1, "route chg/step"),
                   ("rerouted", 1, "flows off default"), ("stretch", 1, "IGP stretch"))


def choose_hysteresis(vgrid, name, trig, hysts, tol_steps=0.125, tol_proxy=0.002, max_stretch=1.02):
    """Shared validation rule, applied identically to every scheme at every trigger:
    eligible = steps above thr <= independent + tol_steps pp, above-thr traffic <= independent + tol_proxy pp,
    IGP stretch <= max_stretch; among eligible take the fewest route changes. None = no setting qualifies."""
    base = summary(vgrid[(name, trig, None)])
    ok = []
    for h in hysts:
        s = summary(vgrid[(name, trig, h)])
        if (s["steps_above_%"] <= base["steps_above_%"] + tol_steps
                and s["above_traffic_%"] <= base["above_traffic_%"] + tol_proxy
                and (np.isnan(s["stretch"]) or s["stretch"] <= max_stretch)):
            ok.append((s["changes_per_step"], s["steps_above_%"], h))
    return min(ok)[2] if ok else None


def _fmt(c, nd=4):
    return f"{c[0]:+.{nd}f} [{c[1]:+.{nd}f}, {c[2]:+.{nd}f}]"


def persistence_report(tgrid, sel, triggers, schemes, boot, block):
    """sel[(scheme, trigger)] = hysteresis chosen on validation (None = independent)."""
    out = {"within_policy": [], "proactive_minus_reactive": []}
    print("\n=== C. Within-policy: persistent MINUS independent, same scheme, same trigger ===")
    print("   (checks the claim 'persistence keeps congestion at or below the independent level' pair by pair)")
    print(f"{'scheme':<10}{'trig':<6}{'h':<6}" + "".join(f"{lab:>30}" for _, _, lab in PERSIST_METRICS[:2] + PERSIST_METRICS[3:4]) + f"{'verdict':>30}")
    for name in schemes:
        if name == "oracle":
            continue
        for trig in triggers:
            h = sel.get((name, trig))
            if h is None:
                print(f"{name:<10}{trig:<6.0%}{'none':<6}  (no persistence setting met the validation rule -> independent routing kept)")
                continue
            p, b = tgrid[(name, trig, h)], tgrid[(name, trig, None)]
            cells = {k: block_ci(sc * (p[k] - b[k]), block, boot) for k, sc, _ in PERSIST_METRICS}
            st, px = cells["above"], cells["proxy"]
            if st[2] <= 0 and px[2] <= 0:
                verdict = "at/below independent (CI)"
            elif st[0] <= 0 and px[0] <= 0:
                verdict = "at/below on average, CI incl. +"
            else:
                verdict = "WORSE than independent (mean)"
            print(f"{name:<10}{trig:<6.0%}{h:<6.2f}" + "".join(f"{_fmt(cells[k]):>30}" for k in ("above", "proxy", "changes")) + f"{verdict:>30}")
            out["within_policy"].append({"scheme": name, "trigger": trig, "hysteresis": h, "verdict": verdict,
                                         **{k: cells[k] for k in cells}})
    print("\n=== D. PERSISTENT proactive MINUS PERSISTENT reactive at the SAME trigger (each with its validation-chosen h) ===")
    print("   negative = forecast better.  Compare with section A (independent) to separate persistence from forecasting.")
    print(f"{'trigger':<9}{'h_react':<9}{'h_pro':<8}" + "".join(f"{lab:>26}" for _, _, lab in PERSIST_METRICS))
    for trig in triggers:
        hr, hp = sel.get(("reactive", trig)), sel.get(("proactive", trig))
        a, b = tgrid[("proactive", trig, hp)], tgrid[("reactive", trig, hr)]
        cells = {k: block_ci(sc * (a[k] - b[k]), block, boot) for k, sc, _ in PERSIST_METRICS}
        f = lambda x: "none" if x is None else f"{x:.2f}"
        print(f"{trig:<9.0%}{f(hr):<9}{f(hp):<8}" + "".join(f"{_fmt(cells[k], 3):>26}" for k, _, _ in PERSIST_METRICS))
        out["proactive_minus_reactive"].append({"trigger": trig, "h_reactive": hr, "h_proactive": hp,
                                                **{k: cells[k] for k in cells}})
    return out


def fixed_report(tgrid, triggers, fixed, boot, block):
    """Section E: proactive minus reactive with the SAME hysteresis for both (removes tuning differences)."""
    print("\n=== E. SAME hysteresis for both schemes: persistent proactive MINUS persistent reactive ===")
    print("   negative = forecast better.  Identical persistence, so differences come from the input matrix only.")
    print(f"{'trigger':<9}{'h':<6}" + "".join(f"{lab:>26}" for _, _, lab in PERSIST_METRICS))
    rows = []
    for h in fixed:
        for trig in triggers:
            a, b = tgrid[("proactive", trig, h)], tgrid[("reactive", trig, h)]
            cells = {k: block_ci(sc * (a[k] - b[k]), block, boot) for k, sc, _ in PERSIST_METRICS}
            print(f"{trig:<9.0%}{h:<6.2f}" + "".join(f"{_fmt(cells[k], 3):>26}" for k, _, _ in PERSIST_METRICS))
            rows.append({"trigger": trig, "hysteresis": h, **cells})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder with true/pred/last_observed _test.npy")
    ap.add_argument("--val-results", default=None, help="folder with true/pred/last_observed _val.npy (trigger selection)")
    ap.add_argument("--nodes", type=int, default=23)
    ap.add_argument("--topo", default="geant")
    ap.add_argument("--edges", default=None)
    ap.add_argument("--link-metadata", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--optimizer", choices=("minmax", "overflow"), default="overflow")
    ap.add_argument("--report-thr", type=float, default=0.70, help="utilisation level the 'above' metrics count")
    ap.add_argument("--triggers", type=float, nargs="+", default=[0.45, 0.55, 0.70],
                    help="controller trigger grid used by EVERY scheme")
    ap.add_argument("--budget", type=float, default=3.0,
                    help="max average flows assigned to non-default paths per step when selecting a trigger; not route changes")
    ap.add_argument("--hybrid", action="store_true", help="also replay the hybrid (max of forecast and last matrix)")
    ap.add_argument("--persistence", action="store_true",
                    help="also replay with route persistence (hysteresis) and compare persistent reactive vs persistent proactive")
    ap.add_argument("--hysteresis", type=float, nargs="+", default=[0.10, 0.30, 0.50, 0.70],
                    help="hysteresis candidates; chosen per scheme and trigger on validation by one shared rule")
    ap.add_argument("--fixed-hysteresis", type=float, nargs="*", default=[],
                    help="sensitivity: also compare persistent proactive vs reactive with ONE hysteresis shared by both "
                         "(e.g. 0.10 0.30), so the two schemes get identical persistence")
    ap.add_argument("--tol-steps", type=float, default=0.125, help="allowed extra steps>thr (pp) vs independent on validation")
    ap.add_argument("--tol-proxy", type=float, default=0.002, help="allowed extra above-thr traffic (pp) vs independent on validation")
    ap.add_argument("--max-stretch", type=float, default=1.02, help="max mean IGP cost stretch on validation")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=96, help="bootstrap block length in steps (96 = one day)")
    ap.add_argument("--out", default=None, help="where to write fair_routing_comparison.json (default: --results)")
    args = ap.parse_args()

    g = build_topology(args.topo, args.nodes, args.edges)
    load_link_metadata(g, args.link_metadata)
    pt = PathTable(g, args.k)
    cap = _capacity_vector(np.array([g[u][v]["capacity_kbps"] for u, v in pt.edges]), pt.E)
    optimize_fn = optimize if args.optimizer == "minmax" else optimize_overflow
    schemes = ["reactive", "proactive", "oracle"] + (["hybrid"] if args.hybrid else [])
    sel_hysts = sorted(args.hysteresis) if args.persistence else []
    hysts = sorted(set(sel_hysts) | set(args.fixed_hysteresis)) if args.persistence else []
    if args.persistence and not args.val_results:
        ap.error("--persistence needs --val-results (the hysteresis must be chosen on validation, not on test)")
    triggers = sorted(args.triggers)
    print(f"{g.number_of_nodes()} nodes, {g.number_of_edges()} links, k={args.k}, optimizer={args.optimizer}; "
          f"'above' metrics count utilisation > {args.report_thr:.0%}; trigger grid {[f'{t:.0%}' for t in triggers]}")

    report = {"config": vars(args)}
    selected = {}
    persist_sel = {}
    if args.val_results:
        val = load_split(args.val_results, "val")
        print(f"\nVALIDATION replay ({len(val[0])} steps)")
        vgrid = run_grid(pt, cap, val, triggers, optimize_fn, args.report_thr, schemes, hysts, args.optimizer)
        print_grid("Validation grid (each scheme at every trigger)", vgrid, triggers, schemes)
        for name in schemes:
            if name == "oracle":
                continue
            trig, feasible = choose_trigger(vgrid, name, triggers, args.budget)
            selected[name] = trig
            print(f"  selected on validation: {name:<9} -> trigger {trig:.0%}"
                  + ("" if feasible else f"   (no trigger met the {args.budget:g} off-default-flows/step budget; took the fewest)"))
        report["validation_selected_triggers"] = selected
        if hysts:
            print("\n  persistence chosen on validation (rule: steps>thr <= independent+%.3fpp, traffic>thr <= independent+%.3fpp, stretch <= %.2f, then fewest route changes)"
                  % (args.tol_steps, args.tol_proxy, args.max_stretch))
            for name in schemes:
                if name == "oracle":
                    continue
                for trig in triggers:
                    h = choose_hysteresis(vgrid, name, trig, sel_hysts, args.tol_steps, args.tol_proxy, args.max_stretch)
                    persist_sel[(name, trig)] = h
                    sv = summary(vgrid[(name, trig, h)])
                    print(f"    {name:<9}@{trig:.0%}: h={'none' if h is None else f'{h:.2f}'}   val route changes/step {sv['changes_per_step']:.3f}")
        report["validation_note"] = ("If pred_val comes from a model fitted on validation (e.g. a stack), the forecast is "
                                     "in-sample there and validation selection favours it.")

    test = load_split(args.results, "test")
    print(f"\nTEST replay ({len(test[0])} steps)")
    tgrid = run_grid(pt, cap, test, triggers, optimize_fn, args.report_thr, schemes, hysts, args.optimizer)
    print_grid("Test grid (each scheme at every trigger)", tgrid, triggers, schemes)
    report["test_grid"] = {f"{n}@{t}@{h}": summary(r) for (n, t, h), r in tgrid.items()}

    pairs = [(t, t) for t in triggers]                       # same trigger for both
    print("\n=== A. SAME trigger for reactive and proactive ===")
    report["same_trigger_diffs"] = print_diffs(tgrid, pairs, args.boot, args.block)
    if selected.get("reactive") is not None and selected.get("proactive") is not None:
        print("\n=== B. Each scheme at the trigger chosen on validation (same rule, same off-default-flow budget) ===")
        sr, sp = selected["reactive"], selected["proactive"]
        for name, trig in (("reactive", sr), ("proactive", sp)):
            s = summary(tgrid[(name, trig, None)])
            print(f"  {name:<9}@{trig:.0%}: steps>thr {s['steps_above_%']:.3f}%  traffic>thr {s['above_traffic_%']:.4f}%  "
                  f"mean peak {s['mean_peak_util']:.3f}  off-default flows {s['flows_off_default_per_step']:.2f}")
        report["validation_selected_diffs"] = print_diffs(tgrid, [(sr, sp)], args.boot, args.block)
    if hysts:
        report["persistence"] = persistence_report(tgrid, persist_sel, triggers,
                                                   [n for n in schemes], args.boot, args.block)
        if args.fixed_hysteresis:
            report["fixed_hysteresis_diffs"] = fixed_report(tgrid, triggers, sorted(args.fixed_hysteresis), args.boot, args.block)
        report["persistence_selected"] = {f"{n}@{t}": h for (n, t), h in persist_sel.items()}
        print("\nEXPLORATORY: the test period was inspected during development, so these intervals describe this one "
              "period and are not confirmatory. Event counts are small; judge by the CI, not the point estimate.")
    print("\nRead it like this: if the interval for 'steps>thr' and 'traffic>thr' includes 0, the forecast made no "
          "demonstrated difference to congestion; off-default flows measure path occupancy, while route changes measure churn.")

    import os
    out = args.out or args.results
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/fair_routing_comparison.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved {out}/fair_routing_comparison.json")


if __name__ == "__main__":
    main()
