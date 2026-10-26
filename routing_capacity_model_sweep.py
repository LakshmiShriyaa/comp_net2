"""Validation-only sweep of path count, proactive trigger, and routing objective."""
import json
from pathlib import Path

import numpy as np

from routing import (_capacity_vector, build_topology, load_link_metadata,
                     PathTable, optimize, optimize_overflow)


CAP_FILE = "geant_link_capacities.csv"
THRESHOLD = 0.70
TRIGGERS = (0.40, 0.45, 0.50, 0.55, 0.60)
PATH_COUNTS = (3, 5, 8)
OBJECTIVES = ("minmax", "overflow")


def replay(pt, actual, route_input, cap, trigger, optimizer):
    optimize_fn = optimize if optimizer == "minmax" else optimize_overflow
    max_util, above, above_proxy, physical, rerouted, stretch = [], [], [], [], [], []
    zero = np.zeros(len(pt.flows), dtype=int)
    for actual_tm, route_tm in zip(actual, route_input):
        demand = pt.demand_vector(route_tm)
        true_demand = pt.demand_vector(actual_tm)
        choice = optimize_fn(pt, demand, cap, trigger)
        load = pt.loads(true_demand, choice)
        util = load / cap
        max_util.append(float(util.max()))
        above.append(float(util.max() > THRESHOLD))
        total = max(float(load.sum()), 1e-9)
        above_proxy.append(float(np.maximum(load - THRESHOLD * cap, 0).sum() / total))
        physical.append(float(np.maximum(load - cap, 0).sum() / total))
        rerouted.append(int((choice != zero).sum()))
        idx = np.arange(len(choice))
        stretch.append(float((true_demand * pt.costs[idx, choice]).sum()
                              / max(float((true_demand * pt.costs[:, 0]).sum()), 1e-9)))
    return {
        "steps_above_70_utilization_%": float(np.mean(above) * 100),
        "above_trigger_traffic_proxy_%": float(np.mean(above_proxy) * 100),
        "physical_overload_traffic_proxy_%": float(np.mean(physical) * 100),
        "mean_max_link_util": float(np.mean(max_util)),
        "p95_max_link_util": float(np.percentile(max_util, 95)),
        "avg_flows_rerouted": float(np.mean(rerouted)),
        "mean_igp_cost_stretch": float(np.mean(stretch)),
    }


def main():
    root = build_topology("geant", 23, "geant_edges.txt")
    load_link_metadata(root, CAP_FILE)
    true_val = np.load("res_STACK_BEST/true_val.npy")
    pred_val = np.load("res_STACK_BEST/pred_val.npy")
    last_val = np.load("res_STACK_BEST/last_observed_val.npy")
    best_candidates = []
    groups = []

    for k in PATH_COUNTS:
        pt = PathTable(root, k=k)
        cap = _capacity_vector(np.array([root[u][v]["capacity_kbps"] for u, v in pt.edges]), pt.E)
        for objective in OBJECTIVES:
            reactive = replay(pt, true_val, last_val, cap, THRESHOLD, objective)
            rows = []
            for trigger in TRIGGERS:
                proactive = replay(pt, true_val, pred_val, cap, trigger, objective)
                row = {"k_paths": k, "objective": objective, "trigger": trigger,
                       **{f"proactive_{key}": value for key, value in proactive.items()},
                       **{f"reactive_{key}": value for key, value in reactive.items()}}
                rows.append(row)
                if (proactive["steps_above_70_utilization_%"] < reactive["steps_above_70_utilization_%"]
                    and proactive["above_trigger_traffic_proxy_%"] <= reactive["above_trigger_traffic_proxy_%"]
                    and proactive["mean_igp_cost_stretch"] <= 1.02):
                    best_candidates.append(row)
            groups.append({"k_paths": k, "objective": objective,
                           "reactive_validation": reactive, "sweep": rows})

    # Prefer fewer high-utilization intervals, then less threshold-excess traffic,
    # then fewer reroutes and less path stretch among validation-qualified options.
    selected = min(best_candidates, key=lambda row: (
        row["proactive_steps_above_70_utilization_%"],
        row["proactive_above_trigger_traffic_proxy_%"],
        row["proactive_avg_flows_rerouted"],
        row["proactive_mean_igp_cost_stretch"],
    )) if best_candidates else None

    output = {
        "split": "validation only",
        "capacity_source": CAP_FILE,
        "path_weights": "archived GÉANT IGP metrics",
        "reactive_trigger": THRESHOLD,
        "trigger_candidates": list(TRIGGERS),
        "path_count_candidates": list(PATH_COUNTS),
        "objectives": list(OBJECTIVES),
        "selection_rule": "qualify if both high-utilization intervals and above-trigger proxy improve versus matched reactive settings and IGP-cost stretch <=1.02; rank by intervals, proxy, reroutes, then stretch",
        "selected": selected,
        "groups": groups,
    }
    Path("routing_capacity_model_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    test = {name: np.load(Path("res_STACK_GEANT_CAP") / f"{name}_test.npy")
            for name in ("true", "pred", "last_observed")}
    test_pt = PathTable(root, k=selected["k_paths"]) if selected else None
    if selected:
        test_cap = _capacity_vector(np.array([root[u][v]["capacity_kbps"] for u, v in test_pt.edges]), test_pt.E)
        reactive_test = replay(test_pt, test["true"], test["last_observed"], test_cap,
                               THRESHOLD, selected["objective"])
        proactive_test = replay(test_pt, test["true"], test["pred"], test_cap,
                                selected["trigger"], selected["objective"])
        test_out = {"split": "test replay; exploratory because test has been inspected during development",
                    "validation_selected": {key: selected[key] for key in ("k_paths", "objective", "trigger")},
                    "reactive": reactive_test, "proactive": proactive_test}
        (Path("res_STACK_GEANT_CAP") / "routing_capacity_model_results.json").write_text(
            json.dumps(test_out, indent=2), encoding="utf-8")
        output["test_replay"] = test_out
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
