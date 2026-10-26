"""Tune persistent-route hysteresis on validation, then replay selected policies."""
import json
from pathlib import Path

import numpy as np

from routing import (_capacity_vector, build_topology, load_link_metadata,
                     PathTable, optimize, optimize_overflow)


TRIGGER = 0.70
REACTIVE_TRIGGER = 0.70
PROACTIVE_TRIGGER = 0.45
HYSTERESIS_CANDIDATES = (0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90)


def optimize_stable(pt, demand, cap, trigger, previous, hysteresis, objective):
    """Keep prior paths unless changing one materially reduces predicted peak use."""
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
        on_worst = np.where((pt.A[np.arange(len(demand)), choice, worst] > 0)
                            & (demand > 0))[0]
        required = (max(1e-6, hysteresis * current_max) if objective == "minmax"
                    else max(1e-12, hysteresis * current_cost))
        best_score = (current_max - required if objective == "minmax"
                      else current_cost - required)
        best_move = None
        for flow in on_worst:
            old = pt.A[flow, choice[flow]]
            for path in range(pt.k):
                if path == choice[flow] or not pt.valid[flow, path]:
                    continue
                new_load = load - demand[flow] * old + demand[flow] * pt.A[flow, path]
                score = (float((new_load / cap).max()) if objective == "minmax"
                         else overflow_cost(new_load))
                if score < best_score:
                    best_score, best_move = score, (flow, path, new_load)
        if best_move is None:
            break
        flow, path, load = best_move
        choice[flow] = path
        current_cost = overflow_cost(load)
    return choice


def replay(pt, true, route_matrix, cap, route_trigger, hysteresis, objective):
    zero = np.zeros(len(pt.flows), dtype=int)
    previous = zero.copy()
    max_util, above, above_proxy, physical_overload = [], [], [], []
    rerouted, changes, stretch = [], [], []
    for actual_tm, route_tm in zip(true, route_matrix):
        demand = pt.demand_vector(route_tm)
        actual = pt.demand_vector(actual_tm)
        if hysteresis is None:
            optimizer = optimize if objective == "minmax" else optimize_overflow
            choice = optimizer(pt, demand, cap, route_trigger)
        else:
            choice = optimize_stable(pt, demand, cap, route_trigger, previous, hysteresis, objective)
        load = pt.loads(actual, choice)
        util = load / cap
        max_util.append(float(util.max()))
        above.append(float(util.max() > TRIGGER))
        total = max(float(load.sum()), 1e-9)
        above_proxy.append(float(np.maximum(load - TRIGGER * cap, 0).sum() / total))
        physical_overload.append(float(np.maximum(load - cap, 0).sum() / total))
        rerouted.append(int((choice != zero).sum()))
        changes.append(int((choice != previous).sum()))
        flow_idx = np.arange(len(choice))
        new_cost = float((actual * pt.costs[flow_idx, choice]).sum())
        base_cost = float((actual * pt.costs[:, 0]).sum())
        stretch.append(new_cost / max(base_cost, 1e-9))
        previous = choice
    return {
        "steps_above_70_utilization_%": float(np.mean(above) * 100),
        "above_trigger_traffic_proxy_%": float(np.mean(above_proxy) * 100),
        "physical_overload_traffic_proxy_%": float(np.mean(physical_overload) * 100),
        "mean_max_link_util": float(np.mean(max_util)),
        "p95_max_link_util": float(np.percentile(max_util, 95)),
        "avg_flows_rerouted": float(np.mean(rerouted)),
        "route_changes_per_interval": float(np.mean(changes)),
        "mean_igp_cost_stretch": float(np.mean(stretch)),
    }


def main():
    results = Path("res_STACK_GEANT_CAP")
    config = json.loads(Path("routing_capacity_model_validation.json").read_text(encoding="utf-8"))["selected"]
    path_count = int(config["k_paths"])
    objective = config["objective"]
    proactive_trigger = float(config["trigger"])
    g = build_topology("geant", 23, "geant_edges.txt")
    load_link_metadata(g, "geant_link_capacities.csv")
    pt = PathTable(g, k=path_count)
    cap = _capacity_vector(np.array([g[u][v]["capacity_kbps"] for u, v in pt.edges]), pt.E)

    validation = {name: np.load(f"res_STACK_BEST/{name}_{split}.npy")
                  for split in ("val",) for name in ("true", "pred", "last_observed")}
    test = {name: np.load(results / f"{name}_test.npy")
            for name in ("true", "pred", "last_observed")}

    modes = {
        "reactive": (validation["last_observed"], test["last_observed"], REACTIVE_TRIGGER),
        "proactive": (validation["pred"], test["pred"], proactive_trigger),
    }
    validation_rows = {}
    selected = {}
    test_results = {}
    for mode, (val_route, test_route, trigger) in modes.items():
        baseline = replay(pt, validation["true"], val_route, cap, trigger, None, objective)
        sweep = []
        for h in HYSTERESIS_CANDIDATES:
            metrics = replay(pt, validation["true"], val_route, cap, trigger, h, objective)
            sweep.append({"hysteresis": h, **metrics})

        # Permit at most 0.125 percentage points more high-utilization intervals
        # and 0.002 percentage points more threshold-excess proxy than independent.
        eligible = [row for row in sweep
                    if row["steps_above_70_utilization_%"] <= baseline["steps_above_70_utilization_%"] + 0.125
                    and row["above_trigger_traffic_proxy_%"] <= baseline["above_trigger_traffic_proxy_%"] + 0.002
                    and row["mean_igp_cost_stretch"] <= 1.02]
        choice = (min(eligible, key=lambda row: (row["route_changes_per_interval"],
                                                 row["steps_above_70_utilization_%"],
                                                 row["above_trigger_traffic_proxy_%"],
                                                 row["hysteresis"])) if eligible else None)
        validation_rows[mode] = {"trigger": trigger, "independent": baseline,
                                 "selected": choice, "selection_tolerance": {
                                     "max_added_steps_above_70_percentage_points": 0.125,
                                     "max_added_above_trigger_proxy_percentage_points": 0.002,
                                     "max_mean_igp_cost_stretch": 1.02},
                                 "sweep": sweep}
        if choice is not None:
            independent_test = replay(pt, test["true"], test_route, cap, trigger, None, objective)
            stable_test = replay(pt, test["true"], test_route, cap, trigger,
                                 choice["hysteresis"], objective)
            test_results[mode] = {"trigger": trigger, "selected_hysteresis": choice["hysteresis"],
                                  "independent": independent_test, "persistent_hysteresis": stable_test}

    validation_report = {
        "split": "validation only",
        "capacity_source": "geant_link_capacities.csv",
        "path_weights": "archived GÉANT IGP metrics",
        "routing_config": {"k_paths": path_count, "objective": objective,
                           "reactive_trigger": REACTIVE_TRIGGER,
                           "proactive_trigger": proactive_trigger},
        "selection_rule": "minimize route changes among positive hysteresis values within stated validation tolerances; tie-break toward the smallest margin",
        "modes": validation_rows,
    }
    test_report = {
        "split": "test replay; exploratory because this period was previously inspected",
        "capacity_source": "geant_link_capacities.csv",
        "path_weights": "archived GÉANT IGP metrics",
        "routing_config": {"k_paths": path_count, "objective": objective,
                           "reactive_trigger": REACTIVE_TRIGGER,
                           "proactive_trigger": proactive_trigger},
        "modes": test_results,
    }
    Path("routing_stability_validation.json").write_text(json.dumps(validation_report, indent=2), encoding="utf-8")
    (results / "routing_stability_results.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")
    print(json.dumps({"validation": validation_report, "test": test_report}, indent=2))


if __name__ == "__main__":
    main()
