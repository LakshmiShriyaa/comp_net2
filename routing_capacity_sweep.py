"""Select a proactive routing trigger on validation with archived link metadata."""
import json
import numpy as np

from routing import build_topology, load_link_metadata, PathTable, evaluate


def main():
    results = "res_STACK_BEST"
    g = build_topology("geant", 23, "geant_edges.txt")
    load_link_metadata(g, "geant_link_capacities.csv")
    pt = PathTable(g, k=3)
    capacity = np.array([g[u][v]["capacity_kbps"] for u, v in pt.edges], dtype=np.float32)
    true = np.load(f"{results}/true_val.npy")
    pred = np.load(f"{results}/pred_val.npy")
    last = np.load(f"{results}/last_observed_val.npy")

    common = dict(capacity=capacity, last_observed=last,
                  forecast_label="Validation-CV-selected per-OD stack (weighted GÉANT paths)")
    reference, _ = evaluate(pt, true, pred, 0.70, 5, 0.9, lag=4, **common)
    reactive = reference["reactive (last matrix)"]
    rows = []
    for trigger in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        scores, _ = evaluate(pt, true, pred, 0.70, 5, 0.9, lag=4,
                             proactive_thr=trigger, **common)
        proactive = scores[f"proactive ({common['forecast_label']})"]
        rows.append({"proactive_trigger": trigger, **{
            f"proactive_{key}": value for key, value in proactive.items()}, **{
            f"reactive_{key}": value for key, value in reactive.items()}})

    eligible = [r for r in rows
                if r["proactive_steps_above_trigger_%"] < reactive["steps_above_trigger_%"]
                and r["proactive_over_trigger_traffic_proxy_%"] <= reactive["over_trigger_traffic_proxy_%"]]
    selected = min(eligible, key=lambda r: (r["proactive_steps_above_trigger_%"],
                                            r["proactive_over_trigger_traffic_proxy_%"],
                                            r["proactive_avg_flows_rerouted"])) if eligible else None
    output = {
        "split": "validation",
        "topology_match": "unique graph isomorphism to archived 2004 GÉANT graph",
        "capacity_source": "geant_link_capacities.csv (kbps)",
        "path_weights": "archived per-link IGP metrics",
        "reactive_trigger": 0.70,
        "selection_rule": "among proactive settings reducing intervals above 70% utilization without increasing the above-trigger traffic proxy, minimize intervals, proxy, then reroutes",
        "reactive_validation": reactive,
        "selected": selected,
        "sweep": rows,
    }
    with open("routing_capacity_validation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
