"""Create presentation-ready charts from a completed result folder.

    python make_presentation_figures.py --results res_FINAL --out presentation
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="res_FINAL")
    ap.add_argument("--out", default="presentation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.results}/metrics.json", encoding="utf-8") as f:
        report = json.load(f)
    test = report["test"]
    display = {
        "Naive (last value)": ("Persistence", "#9aa7b2"),
        "Moving average (3)": ("Moving avg.", "#9aa7b2"),
        "CNN-LSTM": ("CNN–LSTM", "#2878b5"),
        "Per-OD ridge": ("Ridge", "#d89b32"),
        "Flow tree": ("Flow tree", "#8e6bbd"),
        "CNN-LSTM + ridge blend": ("CNN + ridge", "#168a72"),
        "perflow_mean_ridge": ("Per-OD ridge", "#d89b32"),
        "flow_tree": ("Flow tree", "#8e6bbd"),
        "cnn_lstm": ("CNN–LSTM", "#2878b5"),
        "naive_last_value": ("Persistence", "#9aa7b2"),
        "four_way_validation_blend": ("Four-model blend", "#168a72"),
        "perflow_validation_blend": ("Ridge + tree blend", "#168a72"),
        "persistence": ("Persistence", "#9aa7b2"),
        "stack": ("Validation stack", "#168a72"),
    }
    names = [n for n in display if n in test]
    labels = [display[n][0] for n in names]
    colors = [display[n][1] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    x = np.arange(len(names))
    for ax, metric in zip(axes, ("RMSE", "MAE")):
        vals = [test[n]["original_units"][metric] for n in names]
        bars = ax.bar(x, vals, color=colors, width=.66)
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
        ax.set_ylabel(f"{metric} (kbps; lower is better)")
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(0, max(vals) * 1.15)
    axes[0].set_title("RMSE")
    axes[1].set_title("MAE")
    fig.suptitle("GEANT: one-hour-ahead traffic forecast on held-out test data")
    fig.savefig(f"{args.out}/forecast_errors.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    vals = [(test[n]["per_pair_normalized"] if "per_pair_normalized" in test[n] else test[n]["normalized"])["RMSE"] for n in names]
    bars = ax.bar(labels, vals, color=colors, width=.64)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylabel("Per-OD normalized RMSE (lower is better)")
    ax.set_title("Error after normalizing each OD pair by its training range")
    ax.grid(axis="y", alpha=.2)
    ax.set_ylim(0, max(vals) * 1.2)
    fig.savefig(f"{args.out}/normalized_rmse.png", dpi=220)
    plt.close(fig)

    true = np.load(f"{args.results}/true_test.npy")
    pred = np.load(f"{args.results}/pred_test.npy")
    last = np.load(f"{args.results}/last_observed_test.npy")
    n = true.shape[-1]
    offdiag = ~np.eye(n, dtype=bool)
    steps = min(96, len(true))
    actual_sum = true[:steps, offdiag].sum(axis=1) / 1000.0
    pred_sum = pred[:steps, offdiag].sum(axis=1) / 1000.0
    last_sum = last[:steps, offdiag].sum(axis=1) / 1000.0
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.plot(actual_sum, color="#202b35", linewidth=1.5, label="Actual at forecast target")
    ax.plot(pred_sum, color="#168a72", linewidth=1.25, label="Validation-selected forecast")
    ax.plot(last_sum, color="#9aa7b2", linewidth=1.0, alpha=.8, label="Last observed matrix")
    ax.set_xlabel("Test interval (15 minutes each)")
    ax.set_ylabel("Aggregate OD traffic (Mbps)")
    ax.set_title("Forecast example: total GEANT traffic, first 24 hours of the test period")
    ax.grid(alpha=.2)
    ax.legend(ncol=3, fontsize=9)
    fig.savefig(f"{args.out}/forecast_example.png", dpi=220)
    plt.close(fig)

    route_path = f"{args.results}/routing_results.json"
    if os.path.exists(route_path):
        with open(route_path, encoding="utf-8") as f:
            route_report = json.load(f)
            routing = route_report["routing"]
        routes = ["static", "reactive (last matrix)"]
        routes += [r for r in routing if r.startswith("proactive (")]
        routes += [r for r in ("hybrid (max fcst, last)", "oracle (true matrix)") if r in routing]
        routes = list(dict.fromkeys(r for r in routes if r in routing))
        route_label_map = {"static": "Static", "reactive (last matrix)": "Reactive",
                           "hybrid (max fcst, last)": "Hybrid", "oracle (true matrix)": "Oracle"}
        route_labels = [route_label_map.get(r, "Proactive") for r in routes]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
        threshold_metrics = all("steps_above_trigger_%" in routing[r] for r in routes)
        if threshold_metrics:
            congestion = [routing[r]["steps_above_trigger_%"] for r in routes]
            overload = [routing[r]["over_trigger_traffic_proxy_%"] for r in routes]
            threshold = route_report["metadata"]["congestion_threshold"] * 100
            titles = (f"Test intervals above {threshold:.0f}% link utilization",
                      f"Traffic above {threshold:.0f}% utilization proxy")
        else:
            congestion = [routing[r]["steps_with_congestion_%"] for r in routes]
            overload = [routing[r]["overload_traffic_%"] for r in routes]
            titles = ("Test intervals with capacity exceeded", "Overload traffic proxy")
        x = np.arange(len(routes))
        for ax, values, title, color in zip(
                axes, (congestion, overload),
                titles,
                ("#2878b5", "#d89b32")):
            bars = ax.bar(x, values, color=color, width=.66)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
            ax.set_xticks(x, route_labels, rotation=15, ha="right")
            ax.set_ylabel("Percent")
            ax.set_title(title)
            ax.grid(axis="y", alpha=.2)
            ax.set_ylim(0, max(values) * 1.2 + 1e-6)
        if route_report["metadata"].get("link_metadata_file"):
            fig.suptitle("GEANT routing replay (2004 per-link capacities and IGP weights)")
        else:
            fig.suptitle("GEANT routing stress test (assumed uniform link capacity)")
        fig.savefig(f"{args.out}/routing_stress_test.png", dpi=220)
        plt.close(fig)

    stability_path = f"{args.results}/routing_stability_results.json"
    if os.path.exists(stability_path):
        with open(stability_path, encoding="utf-8") as f:
            stability = json.load(f)["modes"]
        labels, scores, colors = [], [], []
        for mode, label, base_color, stable_color in (
                ("reactive", "Reactive", "#2878b5", "#8bbbe0"),
                ("proactive", "Proactive", "#168a72", "#8ac7b6")):
            block = stability[mode]
            labels.extend([label, f"{label}\nstable"])
            scores.extend([block["independent"], block["persistent_hysteresis"]])
            colors.extend([base_color, stable_color])
        panels = (
            ("route_changes_per_interval", "Route changes per interval", False),
            ("steps_above_70_utilization_%", "Intervals above 70% utilization (%)", False),
            ("avg_flows_rerouted", "Flows rerouted per interval", False),
            ("mean_igp_cost_stretch", "Extra IGP path cost (%)", True),
        )
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for ax, (key, title, as_percent) in zip(axes.flat, panels):
            values = [(s[key] - 1.0) * 100 if as_percent else s[key] for s in scores]
            bars = ax.bar(np.arange(len(values)), values, color=colors, width=.68)
            ax.bar_label(bars, fmt="%.3g", padding=2, fontsize=8)
            ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
            ax.set_title(title)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Persistent routing with validation-selected hysteresis")
        fig.savefig(f"{args.out}/routing_stability.png", dpi=220)
        plt.close(fig)

    with open(f"{args.out}/test_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "MAE_kbps", "RMSE_kbps", "MAPE_percent", "Normalized_RMSE"])
        for name in names:
            orig = test[name]["original_units"]
            normalized = test[name]["per_pair_normalized"] if "per_pair_normalized" in test[name] else test[name]["normalized"]
            writer.writerow([name, orig["MAE"], orig["RMSE"], orig["MAPE_%"], normalized["RMSE"]])
    print(f"Wrote charts and test_metrics.csv to {args.out}/")


if __name__ == "__main__":
    main()
