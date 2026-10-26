"""Validation-tuned routing-policy comparison with a hard per-step move cap.

Compares independent overflow proposals and persistent warm-start objectives.
Every controller is limited to at most three final whole-flow route changes per
step. Reactive and forecast-driven inputs share triggers and capacity assumptions.
Tuning uses validation only; the test split is used once for the reported replay.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fair_routing_eval import block_ci, load_split
from routing import (PathTable, _capacity_vector, build_topology,
                     load_link_metadata, optimize_overflow,
                     optimize_overflow_persistent)


def replay(pt, cap, true_matrices, controller_matrices, trigger, report_threshold,
           policy, penalty=0.0, max_moves=100):
    zero = np.zeros(len(pt.flows), dtype=int)
    previous = zero.copy()
    arrays = {k: [] for k in (
        "steps_above", "traffic_proxy", "physical_overload", "peak_util",
        "route_changes", "alternate_flows", "stretch")}
    for actual, controller in zip(true_matrices, controller_matrices):
        demand = pt.demand_vector(controller)
        if policy == "independent_overflow":
            proposed = optimize_overflow(pt, demand, cap, trigger,
                                         max_moves=max_moves)
            choice = bound_independent_changes(
                pt, demand, cap, trigger, previous, proposed, hard_limit=3)
        elif policy == "persistent_overflow":
            choice = optimize_overflow_persistent(
                pt, demand, cap, trigger, previous, penalty,
                max_moves=max_moves, max_changes=3)
        elif policy == "persistent_minmax":
            choice = optimize_overflow_persistent(
                pt, demand, cap, trigger, previous, penalty,
                max_moves=max_moves, objective="minmax", max_changes=3)
        else:
            raise ValueError(f"unknown policy: {policy}")

        true_demand = pt.demand_vector(actual)
        loads = pt.loads(true_demand, choice)
        util = loads / cap
        total = max(float(loads.sum()), 1e-9)
        idx = np.arange(len(choice))
        actual_cost = float((true_demand * pt.costs[idx, choice]).sum())
        shortest_cost = float((true_demand * pt.costs[:, 0]).sum())
        arrays["steps_above"].append(float(util.max() > report_threshold))
        arrays["traffic_proxy"].append(float(np.maximum(
            loads - report_threshold * cap, 0).sum() / total))
        arrays["physical_overload"].append(float(np.maximum(
            loads - cap, 0).sum() / total))
        arrays["peak_util"].append(float(util.max()))
        arrays["route_changes"].append(float(np.count_nonzero(choice != previous)))
        arrays["alternate_flows"].append(float(np.count_nonzero(choice != zero)))
        arrays["stretch"].append(actual_cost / max(shortest_cost, 1e-9))
        previous = choice
    return {k: np.asarray(v, dtype=float) for k, v in arrays.items()}


def bound_independent_changes(pt, demand, cap, trigger, previous, proposed,
                              hard_limit=3):
    """Apply independently proposed routes while enforcing a hard change cap."""
    previous = np.asarray(previous, dtype=int)
    choice = previous.copy()
    load = pt.loads(demand, choice)
    cap = _capacity_vector(cap, pt.E)

    def score(values):
        excess = np.maximum(values / cap - trigger, 0.0)
        return float(excess @ excess)

    current = score(load)
    for _ in range(hard_limit):
        best_score, best_move = current - 1e-12, None
        for f in np.flatnonzero((proposed != choice) & (demand > 0)):
            candidate = load - demand[f] * pt.A[f, choice[f]] + demand[f] * pt.A[f, proposed[f]]
            value = score(candidate)
            if value < best_score:
                best_score, best_move = value, (f, candidate)
        if best_move is None:
            break
        f, load = best_move
        choice[f] = proposed[f]
        current = best_score
    return choice


def summarize(arrays):
    return {
        "steps_above_report_threshold_pct": 100 * float(arrays["steps_above"].mean()),
        "traffic_proxy_above_report_threshold_pct": 100 * float(arrays["traffic_proxy"].mean()),
        "physical_overload_traffic_pct": 100 * float(arrays["physical_overload"].mean()),
        "mean_peak_utilization": float(arrays["peak_util"].mean()),
        "p95_peak_utilization": float(np.percentile(arrays["peak_util"], 95)),
        "route_changes_per_step": float(arrays["route_changes"].mean()),
        "max_route_changes_in_step": int(arrays["route_changes"].max()),
        "alternate_flows_per_step": float(arrays["alternate_flows"].mean()),
        "mean_igp_cost_stretch": float(arrays["stretch"].mean()),
    }


def candidates_for_input(pt, cap, split, input_name, triggers, penalties,
                         report_threshold):
    true, pred, last = split
    controller = pred if input_name == "proactive" else last
    rows = []
    for policy in ("independent_overflow", "persistent_overflow", "persistent_minmax"):
        pgrid = penalties if policy == "persistent_overflow" else [0.0]
        move_grid = [100] if policy == "persistent_overflow" else [1, 3, 10, 100]
        for trigger in triggers:
            for penalty in pgrid:
                for max_moves in move_grid:
                    arrays = replay(pt, cap, true, controller, trigger,
                                    report_threshold, policy, penalty, max_moves)
                    metrics = summarize(arrays)
                    key = (metrics["traffic_proxy_above_report_threshold_pct"],
                           metrics["steps_above_report_threshold_pct"],
                           metrics["mean_peak_utilization"],
                           metrics["route_changes_per_step"],
                           metrics["mean_igp_cost_stretch"], policy, trigger,
                           penalty, max_moves)
                    rows.append({"policy": policy, "trigger": trigger,
                                 "penalty": penalty, "max_moves": max_moves,
                                 "metrics": metrics,
                                 "selection_key": list(key), "arrays": arrays})
    return rows


def paired_differences(a, b, block, boot, seed):
    """Return proactive minus reactive paired block-bootstrap intervals."""
    metrics = {
        "steps_above_report_threshold_pp": ("steps_above", 100.0),
        "traffic_proxy_above_report_threshold_pp": ("traffic_proxy", 100.0),
        "physical_overload_traffic_pp": ("physical_overload", 100.0),
        "mean_peak_utilization": ("peak_util", 1.0),
        "route_changes_per_step": ("route_changes", 1.0),
        "mean_igp_cost_stretch": ("stretch", 1.0),
    }
    out = {}
    for i, (name, (field, scale)) in enumerate(metrics.items()):
        out[name] = dict(zip(("mean", "ci95_low", "ci95_high"),
                             block_ci(scale * (a[field] - b[field]),
                                      block=block, boot=boot, seed=seed + i)))
    return out


def public_candidate(row):
    return {k: v for k, v in row.items() if k not in ("arrays", "selection_key")}


def paired_differences_named(a, b, block, boot, seed):
    return paired_differences(a, b, block, boot, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="res_STACK_GEANT_CAP")
    ap.add_argument("--val-results", default="res_STACK_BEST")
    ap.add_argument("--edges", default="geant_edges.txt")
    ap.add_argument("--link-metadata", default="geant_link_capacities.csv")
    ap.add_argument("--triggers", type=float, nargs="+", default=[0.45, 0.55, 0.70])
    ap.add_argument("--penalties", type=float, nargs="+",
                    default=[0.0, 0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--report-threshold", type=float, default=0.70)
    ap.add_argument("--budget", type=float, default=3.0,
                    help="maximum average whole-flow route changes per step, selected on validation")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.50, 0.35])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--out", default="res_STACK_GEANT_CAP/policy_objective_sweep.json")
    args = ap.parse_args()
    if args.budget <= 0 or any(x <= 0 for x in args.scales):
        ap.error("budget and all capacity scales must be positive")

    graph = build_topology("geant", 23, args.edges)
    load_link_metadata(graph, args.link_metadata)
    pt = PathTable(graph, k=5)
    base_cap = _capacity_vector(
        np.array([graph[u][v]["capacity_kbps"] for u, v in pt.edges]), pt.E)
    validation = load_split(args.val_results, "val")
    test = load_split(args.results, "test")
    report = {"setup": vars(args), "scenarios": {}}

    for scale in args.scales:
        cap = base_cap * scale
        scenario = {"capacity_scale": scale,
                    "capacity_note": "1.0 uses archived capacities; other scales are synthetic stress scenarios",
                    "average_route_change_budget": args.budget,
                    "hard_route_change_cap_per_step": 3,
                    "validation_selection": {}, "test": {}, "paired_test_differences": {}}
        candidates = {}
        for input_name in ("reactive", "proactive"):
            candidates[input_name] = candidates_for_input(
                pt, cap, validation, input_name, sorted(args.triggers),
                sorted(args.penalties), args.report_threshold)

        keyed = {}
        for input_name in ("reactive", "proactive"):
            keyed[input_name] = {
                (r["policy"], r["trigger"], r["penalty"], r["max_moves"]): r
                for r in candidates[input_name]}
        common = []
        for key in set(keyed["reactive"]) & set(keyed["proactive"]):
            r, p = keyed["reactive"][key], keyed["proactive"][key]
            mr, mp = r["metrics"], p["metrics"]
            budget_met = (mr["route_changes_per_step"] <= args.budget and
                          mp["route_changes_per_step"] <= args.budget)
            score = (
                (mr["traffic_proxy_above_report_threshold_pct"] +
                 mp["traffic_proxy_above_report_threshold_pct"]) / 2,
                (mr["steps_above_report_threshold_pct"] +
                 mp["steps_above_report_threshold_pct"]) / 2,
                max(mr["route_changes_per_step"], mp["route_changes_per_step"]),
                (mr["mean_peak_utilization"] + mp["mean_peak_utilization"]) / 2,
                (mr["mean_igp_cost_stretch"] + mp["mean_igp_cost_stretch"]) / 2,
            )
            common.append({"key": key, "reactive": r, "proactive": p,
                           "budget_met": budget_met, "score": score})
        eligible = [row for row in common if row["budget_met"]]
        if eligible:
            picked_common = min(eligible, key=lambda row: row["score"] + row["key"])
        else:
            picked_common = min(common, key=lambda row: (
                max(row["reactive"]["metrics"]["route_changes_per_step"],
                    row["proactive"]["metrics"]["route_changes_per_step"]),
                row["score"], row["key"]))
        policy, trigger, penalty, max_moves = picked_common["key"]
        scenario["validation_selection"] = {
            "selected_common_controller": {
                "policy": policy, "trigger": trigger, "penalty": penalty,
                "independent_max_moves": max_moves,
                "budget_met_for_both_inputs": picked_common["budget_met"],
                "selection_score": list(picked_common["score"]),
                "reactive_metrics": picked_common["reactive"]["metrics"],
                "proactive_metrics": picked_common["proactive"]["metrics"],
            },
            "candidate_pairs": [
                {"policy": row["key"][0], "trigger": row["key"][1],
                 "penalty": row["key"][2], "max_moves": row["key"][3],
                 "budget_met_for_both_inputs": row["budget_met"],
                 "selection_score": list(row["score"]),
                 "reactive_metrics": row["reactive"]["metrics"],
                 "proactive_metrics": row["proactive"]["metrics"]}
                for row in sorted(common, key=lambda row: row["score"] + row["key"])],
        }

        true, pred, last = test
        baseline_rows = [r for r in candidates["reactive"]
                         if r["policy"] == "independent_overflow" and
                         r["trigger"] == trigger]
        baseline_eligible = [r for r in baseline_rows
                             if r["metrics"]["route_changes_per_step"] <= args.budget]
        if baseline_eligible:
            baseline_pick = min(baseline_eligible, key=lambda r: tuple(r["selection_key"]))
        else:
            baseline_pick = min(baseline_rows, key=lambda r: (
                r["metrics"]["route_changes_per_step"], tuple(r["selection_key"])))

        baseline_r = replay(pt, cap, true, last, trigger, args.report_threshold,
                            "independent_overflow", 0, baseline_pick["max_moves"])
        baseline_p = replay(pt, cap, true, pred, trigger, args.report_threshold,
                            "independent_overflow", 0, baseline_pick["max_moves"])
        tuned_r = replay(pt, cap, true, last, trigger, args.report_threshold,
                         policy, penalty, max_moves)
        tuned_p = replay(pt, cap, true, pred, trigger, args.report_threshold,
                         policy, penalty, max_moves)
        test_arrays = {"reactive_independent_baseline": baseline_r,
                       "proactive_independent": baseline_p,
                       "reactive_tuned_policy": tuned_r,
                       "proactive_tuned_policy": tuned_p}
        scenario["validation_selection"]["reactive_independent_baseline"] = {
            "trigger": trigger, "max_moves_per_optimizer_call": baseline_pick["max_moves"],
            "average_route_change_budget_met": baseline_pick["metrics"]["route_changes_per_step"] <= args.budget,
            "validation_metrics": baseline_pick["metrics"],
        }
        scenario["test"] = {
            name: {**summarize(arrays),
                   "validation_selected_trigger": trigger,
                   "average_route_change_budget_met":
                       summarize(arrays)["route_changes_per_step"] <= args.budget}
            for name, arrays in test_arrays.items()}
        scenario["paired_test_differences"] = {
            "reactive_tuned_policy_minus_reactive_independent_baseline":
                paired_differences(tuned_r, baseline_r, args.block, args.boot, 7000 + int(scale * 100)),
            "proactive_tuned_policy_minus_reactive_tuned_policy":
                paired_differences(tuned_p, tuned_r, args.block, args.boot, 8000 + int(scale * 100)),
            "proactive_tuned_policy_minus_reactive_independent_baseline":
                paired_differences(tuned_p, baseline_r, args.block, args.boot, 9000 + int(scale * 100)),
            "proactive_independent_minus_reactive_independent_baseline":
                paired_differences(baseline_p, baseline_r, args.block, args.boot, 10000 + int(scale * 100)),
        }
        report["scenarios"][str(scale)] = scenario
        print(f"scale={scale:g}; common policy={policy}; trigger={trigger:.0%}; "
              f"penalty={penalty:g}; independent max-moves={baseline_pick['max_moves']}")
        for name, metrics in scenario["test"].items():
            print(f"  {name}: {metrics}")
        print(f"  forecast paired delta: "
              f"{scenario['paired_test_differences']['proactive_tuned_policy_minus_reactive_tuned_policy']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Validation-selected routing policy comparison", "",
          "Test period was inspected during development; these results are exploratory. "
          "All controllers use the same link capacities and trigger, and are limited to three final whole-flow route changes per step. "
          "The capacity scales below 1.0 are synthetic stress scenarios.", ""]
    for scale, scenario in report["scenarios"].items():
        selected = scenario["validation_selection"]["selected_common_controller"]
        md += [f"## Capacity scale {float(scale):g}", "",
               f"Validation selected {selected['policy']} at a {selected['trigger']:.0%} trigger "
               f"(penalty {selected['penalty']:g}); hard route-change cap is 3 per step.", "",
               "| Test policy | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | Alternate flows/step | IGP stretch |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for name, m in scenario["test"].items():
            md.append(f"| {name} | {m['steps_above_report_threshold_pct']:.3f}% | "
                      f"{m['traffic_proxy_above_report_threshold_pct']:.6f}% | "
                      f"{m['mean_peak_utilization']:.3f} | {m['route_changes_per_step']:.3f} | "
                      f"{m['alternate_flows_per_step']:.2f} | {m['mean_igp_cost_stretch']:.4f} |")
        md += ["", "Paired proactive-minus-reactive differences and 95% moving-block bootstrap intervals:", "",
               "```json", json.dumps(scenario["paired_test_differences"], indent=2), "```", ""]
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {md_path}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
