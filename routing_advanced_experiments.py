"""Validation-tune reroute-penalized and splittable MCF routing policies.

All methods use the same k-shortest-path set, per-link capacities, action
threshold, and report threshold. MCF splits each OD demand fractionally and is
therefore a capacity-aware relaxation/ECMP-style benchmark, not a single-path
per-flow policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from fair_routing_eval import block_ci, load_split
from routing import (PathTable, _capacity_vector, build_topology,
                     load_link_metadata, optimize_overflow)


def optimize_overflow_with_reroute_cost(pt, demand, cap, threshold, previous,
                                        penalty, max_moves=100):
    """Reduce threshold overflow while penalizing changes from last interval."""
    demand = np.maximum(np.asarray(demand, dtype=np.float32), 0)
    cap = _capacity_vector(cap, pt.E)
    choice = np.asarray(previous, dtype=int).copy()
    load = pt.loads(demand, choice)

    def objective(values, routes):
        excess = np.maximum(values / cap - threshold, 0.0)
        return float(excess @ excess + penalty * np.count_nonzero(routes != previous))

    current = objective(load, choice)
    flow_idx = np.arange(len(demand))
    for _ in range(max_moves):
        util = load / cap
        if util.max() <= threshold:
            break
        worst = int(util.argmax())
        active = np.where((pt.A[flow_idx, choice, worst] > 0) & (demand > 0))[0]
        best_score, best_move = current - 1e-12, None
        for f in active:
            old_path = pt.A[f, choice[f]]
            for p in range(pt.k):
                if p == choice[f] or not pt.valid[f, p]:
                    continue
                candidate_load = load - demand[f] * old_path + demand[f] * pt.A[f, p]
                # The candidate differs from current routes for exactly this flow.
                route_delta = int(p != previous[f]) - int(choice[f] != previous[f])
                excess = np.maximum(candidate_load / cap - threshold, 0.0)
                score = float(excess @ excess + penalty *
                              (np.count_nonzero(choice != previous) + route_delta))
                if score < best_score:
                    best_score, best_move = score, (f, p, candidate_load)
        if best_move is None:
            break
        f, p, load = best_move
        choice[f] = p
        current = best_score
    return choice


def optimize_split_mcf(pt, demand, cap, threshold, path_penalty, lp_data):
    """Minimize max capacity utilization over fractional paths plus IGP stretch cost."""
    demand = np.maximum(np.asarray(demand, dtype=np.float64), 0)
    cap = np.asarray(cap, dtype=np.float64)
    F, K, E = pt.A.shape
    shortest = np.zeros(F, dtype=int)
    static_load = pt.loads(demand, shortest)
    if float((static_load / cap).max()) <= threshold:
        out = np.zeros((F, K), dtype=np.float64)
        out[:, 0] = 1.0
        return out

    total = max(float(demand.sum()), 1e-12)
    c = np.zeros(F * K + 1, dtype=np.float64)
    c[-1] = 1.0  # maximum utilization variable
    relative_extra_cost = np.maximum(pt.costs / np.maximum(pt.costs[:, :1], 1e-12) - 1, 0)
    c[:F * K] = (path_penalty * demand[:, None] / total * relative_extra_cost).reshape(-1)

    # Link-load constraints: sum_f,p demand_f * fraction_fp * incidence_fpe <= cap_e * u.
    fs, ps, es = lp_data["incidence"]
    values = demand[fs]
    A_ub = coo_matrix(
        (np.r_[values, -cap],
         (np.r_[es, np.arange(E)], np.r_[fs * K + ps, np.full(E, F * K)])),
        shape=(E, F * K + 1)).tocsr()
    result = linprog(c, A_ub=A_ub, b_ub=np.zeros(E), A_eq=lp_data["A_eq"],
                     b_eq=np.ones(F), bounds=lp_data["bounds"], method="highs")
    if not result.success:
        raise RuntimeError(f"MCF LP failed: {result.message}")
    return result.x[:F * K].reshape(F, K)


def make_lp_data(pt):
    F, K, _ = pt.A.shape
    fs, ps, es = np.nonzero(pt.A)
    eq_rows = np.repeat(np.arange(F), K)
    eq_cols = np.arange(F * K)
    A_eq = coo_matrix((np.ones(F * K), (eq_rows, eq_cols)),
                      shape=(F, F * K + 1)).tocsr()
    bounds = [(0.0, 1.0 if pt.valid[f, p] else 0.0)
              for f in range(F) for p in range(K)]
    bounds.append((0.0, None))
    return {"incidence": (fs, ps, es), "A_eq": A_eq, "bounds": bounds}


def route_episode(pt, cap, matrices, actual, threshold, report_threshold,
                  policy, penalty=0.0, mcf_penalty=0.0, lp_data=None):
    zero = np.zeros(len(pt.flows), dtype=int)
    previous = zero.copy()
    previous_fraction = None
    max_util, above, above_proxy, physical, changes, alt, stretch = [], [], [], [], [], [], []
    for route_tm, true_tm in zip(matrices, actual):
        demand = pt.demand_vector(route_tm)
        true_demand = pt.demand_vector(true_tm)
        if policy == "greedy_overflow":
            choice = optimize_overflow(pt, demand, cap, threshold)
            fractions = None
        elif policy == "reroute_cost":
            choice = optimize_overflow_with_reroute_cost(
                pt, demand, cap, threshold, previous, penalty)
            fractions = None
        elif policy == "mcf_split":
            fractions = optimize_split_mcf(pt, demand, cap, threshold, mcf_penalty, lp_data)
            choice = None
        else:
            raise ValueError(policy)

        if fractions is None:
            load = pt.loads(true_demand, choice)
            changes.append(int(np.count_nonzero(choice != previous)))
            alt.append(int(np.count_nonzero(choice != 0)))
            idx = np.arange(len(choice))
            route_cost = float((true_demand * pt.costs[idx, choice]).sum())
            previous = choice
        else:
            load = np.einsum("f,fp,fpe->e", true_demand, fractions, pt.A)
            if previous_fraction is None:
                base = np.zeros_like(fractions)
                base[:, 0] = 1.0
                changes.append(float(0.5 * np.abs(fractions - base).sum()))
            else:
                changes.append(float(0.5 * np.abs(fractions - previous_fraction).sum()))
            alt.append(float(fractions[:, 1:].sum()))
            route_cost = float(np.sum(true_demand[:, None] * fractions * pt.costs))
            previous_fraction = fractions

        util = load / cap
        total = max(float(load.sum()), 1e-9)
        max_util.append(float(util.max()))
        above.append(float(util.max() > report_threshold))
        above_proxy.append(float(np.maximum(load - report_threshold * cap, 0).sum() / total))
        physical.append(float(np.maximum(load - cap, 0).sum() / total))
        base_cost = float((true_demand * pt.costs[:, 0]).sum())
        stretch.append(route_cost / max(base_cost, 1e-9))
    return {
        "steps_above_report_threshold_pct": 100 * float(np.mean(above)),
        "above_threshold_traffic_proxy_pct": 100 * float(np.mean(above_proxy)),
        "above_capacity_traffic_proxy_pct": 100 * float(np.mean(physical)),
        "mean_peak_utilization": float(np.mean(max_util)),
        "route_changes_or_split_equivalents_per_step": float(np.mean(changes)),
        "flows_using_nonshortest_paths_per_step": float(np.mean(alt)),
        "mean_igp_cost_stretch": float(np.mean(stretch)),
    }


def tune_penalty(pt, cap, actual, route, trigger, report_threshold, penalties,
                 budget, kind, lp_data):
    rows = []
    for value in penalties:
        metrics = route_episode(pt, cap, route, actual, trigger, report_threshold,
                                kind, penalty=value if kind == "reroute_cost" else 0,
                                mcf_penalty=value if kind == "mcf_split" else 0,
                                lp_data=lp_data)
        rows.append({"penalty": value, **metrics})
        print(f"    tuned {kind}: penalty={value:g}, "
              f"moves={metrics['route_changes_or_split_equivalents_per_step']:.2f}/step",
              flush=True)
    eligible = [row for row in rows
                if row["route_changes_or_split_equivalents_per_step"] <= budget]
    selected = min(eligible, key=lambda x: (x["above_threshold_traffic_proxy_pct"],
                                            x["steps_above_report_threshold_pct"],
                                            x["route_changes_or_split_equivalents_per_step"],
                                            x["penalty"])) if eligible else min(
        rows, key=lambda x: (x["route_changes_or_split_equivalents_per_step"],
                             x["above_threshold_traffic_proxy_pct"]))
    return selected, bool(eligible), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="res_STACK_GEANT_CAP")
    ap.add_argument("--val-results", default="res_STACK_BEST")
    ap.add_argument("--nodes", type=int, default=23)
    ap.add_argument("--edges", default="geant_edges.txt")
    ap.add_argument("--link-metadata", default="geant_link_capacities.csv")
    ap.add_argument("--trigger", type=float, default=0.55,
                    help="common action threshold for every policy (default 55%, selected previously on validation)")
    ap.add_argument("--report-threshold", type=float, default=0.70)
    ap.add_argument("--budget", type=float, default=3.0,
                    help="maximum average route changes/split-flow equivalents per step for validation selection")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.35],
                    help="capacity multipliers; 1.0 is archived capacities, smaller values are synthetic stress cases")
    ap.add_argument("--reroute-penalties", type=float, nargs="+",
                    default=[0.0, 0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--mcf-penalties", type=float, nargs="+",
                    default=[0.01, 0.1])
    ap.add_argument("--out", default="res_STACK_GEANT_CAP/advanced_routing")
    args = ap.parse_args()
    if any(x <= 0 for x in args.scales):
        ap.error("all capacity scales must be > 0")

    graph = build_topology("geant", args.nodes, args.edges)
    load_link_metadata(graph, args.link_metadata)
    pt = PathTable(graph, k=5)
    base_cap = _capacity_vector(np.array([graph[u][v]["capacity_kbps"] for u, v in pt.edges]), pt.E)
    lp_data = make_lp_data(pt)
    report = {"setup": vars(args), "scenarios": {}}

    for scale in args.scales:
        cap = base_cap * scale
        scenario = {"capacity_scale": scale,
                    "capacity_note": "scale 1.0 uses archived link capacities; other scales are synthetic sensitivity cases",
                    "trigger": args.trigger, "report_threshold": args.report_threshold,
                    "validation_tuning": {}, "validation": {}, "test": {}}
        for split_name, folder in (("validation", args.val_results), ("test", args.results)):
            true, pred, last = load_split(folder, "val" if split_name == "validation" else "test")
            route_inputs = {"reactive": last, "proactive": pred}
            target = true
            split_results = {}
            for input_name, route_matrices in route_inputs.items():
                # Standard independent greedy overflow reference at the common trigger.
                split_results[f"{input_name}_greedy_overflow"] = route_episode(
                    pt, cap, route_matrices, target, args.trigger, args.report_threshold,
                    "greedy_overflow", lp_data=lp_data)

                # Tune a single route-change penalty per input type on validation only.
                if split_name == "validation":
                    selected, feasible, sweep = tune_penalty(
                        pt, cap, target, route_matrices, args.trigger, args.report_threshold,
                        args.reroute_penalties, args.budget, "reroute_cost", lp_data)
                    scenario["validation_tuning"][f"{input_name}_reroute_cost"] = {
                        "selected": selected, "budget_met": feasible, "sweep": sweep}
                    split_results[f"{input_name}_reroute_cost"] = selected
                else:
                    val_choice = scenario["validation_tuning"].get(f"{input_name}_reroute_cost")
                    if val_choice is None:
                        raise RuntimeError("reroute-cost penalty must be selected on validation before test")
                    penalty = val_choice["selected"]["penalty"]
                    split_results[f"{input_name}_reroute_cost"] = {"selected_penalty": penalty, **route_episode(
                        pt, cap, route_matrices, target, args.trigger, args.report_threshold,
                        "reroute_cost", penalty=penalty, lp_data=lp_data)}

                # Tune MCF's path-cost regularizer on validation, then test at that setting.
                if split_name == "validation":
                    selected_mcf, feasible_mcf, sweep_mcf = tune_penalty(
                        pt, cap, target, route_matrices, args.trigger, args.report_threshold,
                        args.mcf_penalties, args.budget, "mcf_split", lp_data)
                    scenario["validation_tuning"][f"{input_name}_mcf_split"] = {
                        "selected": selected_mcf, "budget_met": feasible_mcf, "sweep": sweep_mcf}
                    split_results[f"{input_name}_mcf_split"] = selected_mcf
                else:
                    penalty = scenario["validation_tuning"]["mcf_joint"]["selected_penalty"]
                    split_results[f"{input_name}_mcf_split"] = {"selected_penalty": penalty, **route_episode(
                        pt, cap, route_matrices, target, args.trigger, args.report_threshold,
                        "mcf_split", mcf_penalty=penalty, lp_data=lp_data)}
            if split_name == "validation":
                # Use one MCF path-cost coefficient for both reactive and forecast inputs.
                # This keeps the routing objective identical when comparing the forecasts.
                reactive_rows = {row["penalty"]: row for row in
                                 scenario["validation_tuning"]["reactive_mcf_split"]["sweep"]}
                proactive_rows = {row["penalty"]: row for row in
                                  scenario["validation_tuning"]["proactive_mcf_split"]["sweep"]}
                joint_rows = []
                for penalty in sorted(set(reactive_rows) & set(proactive_rows)):
                    r, f = reactive_rows[penalty], proactive_rows[penalty]
                    joint_rows.append({
                        "penalty": penalty,
                        "mean_above_threshold_traffic_proxy_pct":
                            (r["above_threshold_traffic_proxy_pct"] + f["above_threshold_traffic_proxy_pct"]) / 2,
                        "max_route_changes_or_split_equivalents_per_step":
                            max(r["route_changes_or_split_equivalents_per_step"],
                                f["route_changes_or_split_equivalents_per_step"]),
                        "mean_steps_above_report_threshold_pct":
                            (r["steps_above_report_threshold_pct"] + f["steps_above_report_threshold_pct"]) / 2,
                    })
                joint_feasible = [r for r in joint_rows
                                  if r["max_route_changes_or_split_equivalents_per_step"] <= args.budget]
                joint_selected = (min(joint_feasible, key=lambda r: (
                    r["mean_above_threshold_traffic_proxy_pct"],
                    r["mean_steps_above_report_threshold_pct"],
                    r["max_route_changes_or_split_equivalents_per_step"], r["penalty"]))
                    if joint_feasible else min(joint_rows, key=lambda r: (
                        r["max_route_changes_or_split_equivalents_per_step"],
                        r["mean_above_threshold_traffic_proxy_pct"])))
                scenario["validation_tuning"]["mcf_joint"] = {
                    "selected_penalty": joint_selected["penalty"],
                    "budget_met": bool(joint_feasible),
                    "selection": joint_selected,
                    "sweep": joint_rows,
                }
                scenario["validation"]["mcf_joint_selection"] = scenario["validation_tuning"]["mcf_joint"]
            scenario[split_name] = split_results
        report["scenarios"][str(scale)] = scenario
        print(f"Completed capacity scale {scale:g}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "advanced_routing_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out / 'advanced_routing_results.json'}")


if __name__ == "__main__":
    main()
