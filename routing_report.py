"""Export routing replay JSON as compact CSV and Markdown tables."""
import csv
import json
from pathlib import Path


ROOT = Path("res_STACK_GEANT_CAP")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value):
    return f"{value:.3f}%"


def make_route_rows(report):
    metadata = report["metadata"]
    rows = []
    for label, values in report["routing"].items():
        selected_threshold = (metadata["proactive_threshold"]
                              if label.startswith(("proactive", "hybrid"))
                              else metadata["congestion_threshold"])
        rows.append({
            "policy": label,
            "controller_threshold": selected_threshold,
            "steps_above_controller_threshold_pct": values["steps_above_trigger_%"],
            "above_controller_threshold_traffic_proxy_pct": values["over_trigger_traffic_proxy_%"],
            "mean_max_link_utilization": values["mean_max_link_util"],
            "p95_max_link_utilization": values["p95_max_link_util"],
            "flows_rerouted_per_interval": values["avg_flows_rerouted"],
            "mean_igp_cost_stretch": values["path_stretch"],
            "physical_overload_traffic_proxy_pct": values["overload_traffic_%"],
        })
    return rows


def make_stability_rows(report):
    rows = []
    for mode, values in report["modes"].items():
        for kind, key in (("Independent", "independent"),
                          (f"Persistent (h={values['selected_hysteresis']:.3g})", "persistent_hysteresis")):
            metric = values[key]
            rows.append({
                "input_policy": mode,
                "action_threshold": values["trigger"],
                "route_handling": kind,
                "steps_above_70_utilization_pct": metric["steps_above_70_utilization_%"],
                "above_70_utilization_proxy_pct": metric["above_trigger_traffic_proxy_%"],
                "route_changes_per_interval": metric["route_changes_per_interval"],
                "flows_rerouted_per_interval": metric["avg_flows_rerouted"],
                "mean_igp_cost_stretch": metric["mean_igp_cost_stretch"],
            })
    return rows


def markdown_table(rows, columns):
    head = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        cells = []
        for _, field in columns:
            value = row[field]
            if field.endswith("_pct"):
                cells.append(pct(value))
            elif field in ("controller_threshold", "action_threshold"):
                cells.append(f"{value:.0%}")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def main():
    main_report = load(ROOT / "routing_results.json")
    stability = load(ROOT / "routing_stability_results.json")
    route_rows = make_route_rows(main_report)
    stability_rows = make_stability_rows(stability)

    route_fields = list(route_rows[0])
    stability_fields = list(stability_rows[0])
    write_csv(ROOT / "routing_comparison.csv", route_rows, route_fields)
    write_csv(ROOT / "routing_stability_comparison.csv", stability_rows, stability_fields)

    reactive = main_report["routing"]["reactive (last matrix)"]
    proactive_label = next(k for k in main_report["routing"] if k.startswith("proactive ("))
    proactive = main_report["routing"][proactive_label]
    trigger = main_report["metadata"]["congestion_threshold"]
    proactive_trigger = main_report["metadata"]["proactive_threshold"]
    same_trigger = abs(trigger - proactive_trigger) < 1e-9
    detection = main_report["detection"]
    forecast_name = next(k for k in detection if "forecast" in k.lower())
    last_name = next(k for k in detection if "last observed" in k.lower())

    lines = [
        "# Routing replay summary", "",
        "Test replay on archived GÉANT per-link capacities and IGP weights. "
        "The test period has been inspected during development; treat these results as exploratory.", "",
        f"Configuration: **{main_report['metadata']['k_paths']} paths, "
        f"{main_report['metadata']['routing_objective']} objective; reactive/oracle threshold "
        f"{trigger:.0%}, proactive threshold {proactive_trigger:.0%}**. "
        + ("All routing controllers use the same threshold in this comparison." if same_trigger else
           "Thresholds differ across controllers; compare this as a tuned-policy result, not an isolated forecast effect."), "",
        f"No policy exceeded full link capacity in this replay. The table reports utilization above the "
        f"chosen {trigger:.0%} controller threshold; that threshold is an operating choice, not physical congestion.", "",
        markdown_table(route_rows, [
            ("Policy", "policy"), ("Trigger", "controller_threshold"),
            (f"Steps >{trigger:.0%}", "steps_above_controller_threshold_pct"),
            (f"Traffic proxy >{trigger:.0%}", "above_controller_threshold_traffic_proxy_pct"),
            ("Mean peak util", "mean_max_link_utilization"), ("P95 peak util", "p95_max_link_utilization"),
            ("Flows rerouted / step", "flows_rerouted_per_interval"),
            ("IGP cost stretch", "mean_igp_cost_stretch"),
        ]), "",
        "## Link-load estimation at a common threshold", "",
        markdown_table([
            {"source": last_name, **detection[last_name]},
            {"source": forecast_name, **detection[forecast_name]},
        ], [
            ("Input", "source"), ("Link-load RMSE (% capacity)", "link_load_RMSE_%cap"),
            ("Precision", "precision"), ("Recall", "recall"), ("F1", "F1"),
        ]), "",
        "The forecast's link-load RMSE is lower on the same targets and links. Detection scores use the same "
        f"{trigger:.0%} threshold for truth and estimates; they are threshold-specific.", "",
        "## Persistent route stability comparison", "",
        markdown_table(stability_rows, [
            ("Input", "input_policy"), ("Action threshold", "action_threshold"), ("Handling", "route_handling"),
            ("Steps >70%", "steps_above_70_utilization_pct"),
            ("Traffic proxy >70%", "above_70_utilization_proxy_pct"),
            ("Route changes / step", "route_changes_per_interval"),
            ("Flows rerouted / step", "flows_rerouted_per_interval"),
            ("IGP cost stretch", "mean_igp_cost_stretch"),
        ]), "",
        "Persistent-route rows use the separately validation-selected threshold for each input type; compare independent vs persistent within each input. Utilization results do not establish packet-loss, delay, or throughput improvement.",
    ]
    (ROOT / "routing_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {ROOT / 'routing_summary.md'}")
    print(f"Wrote {ROOT / 'routing_comparison.csv'}")
    print(f"Wrote {ROOT / 'routing_stability_comparison.csv'}")


if __name__ == "__main__":
    main()
