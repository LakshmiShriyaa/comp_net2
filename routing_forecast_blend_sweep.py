"""Validation-tune forecast/last-observation blending for persistent routing."""
import argparse
import json
from pathlib import Path

import numpy as np

from fair_routing_eval import block_ci, load_split
from routing import PathTable, _capacity_vector, build_topology, load_link_metadata
from routing_policy_fair_eval import replay, summarize


def differences(a, b, block, boot, seed):
    fields = {
        "steps_above_70_pp": ("steps_above", 100),
        "traffic_proxy_above_70_pp": ("traffic_proxy", 100),
        "physical_overload_pp": ("physical_overload", 100),
        "mean_peak_utilization": ("peak_util", 1),
        "route_changes_per_step": ("route_changes", 1),
        "mean_igp_cost_stretch": ("stretch", 1),
    }
    out = {}
    for i, (label, (field, scale)) in enumerate(fields.items()):
        mean, lo, hi = block_ci(scale * (a[field] - b[field]), block, boot, seed + i)
        out[label] = {"mean": mean, "ci95_low": lo, "ci95_high": hi}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="res_STACK_GEANT_CAP")
    ap.add_argument("--val-results", default="res_STACK_BEST")
    ap.add_argument("--edges", default="geant_edges.txt")
    ap.add_argument("--link-metadata", default="geant_link_capacities.csv")
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                    help="forecast weight; 0=reactive last matrix, 1=full forecast")
    ap.add_argument("--budget", type=float, default=3.0,
                    help="average whole-flow path changes per step")
    ap.add_argument("--report-threshold", type=float, default=0.70)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.50, 0.35])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--policy-results", nargs="+", default=[
        "res_STACK_GEANT_CAP/policy_fair_results_real.json",
        "res_STACK_GEANT_CAP/policy_fair_results_stress.json"])
    ap.add_argument("--out", default="res_STACK_GEANT_CAP/forecast_blend_results.json")
    args = ap.parse_args()

    selected_controls = {}
    for filename in args.policy_results:
        path = Path(filename)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for scale, scenario in data["scenarios"].items():
                x = scenario["validation_selection"]["selected_common_controller"]
                selected_controls[str(float(scale))] = {
                    "trigger": x["trigger"], "penalty": x["penalty"],
                    "policy": x["policy"],
                }

    graph = build_topology("geant", 23, args.edges)
    load_link_metadata(graph, args.link_metadata)
    pt = PathTable(graph, k=5)
    base_cap = _capacity_vector(
        np.array([graph[u][v]["capacity_kbps"] for u, v in pt.edges]), pt.E)
    val = load_split(args.val_results, "val")
    test = load_split(args.results, "test")
    report = {"setup": vars(args), "scenarios": {}}

    for scale in args.scales:
        control = selected_controls.get(str(float(scale)))
        if control is None:
            raise ValueError(f"No validation-selected controller settings for capacity scale {scale:g}")
        cap = base_cap * scale
        trigger, penalty = control["trigger"], control["penalty"]
        val_true, val_pred, val_last = val
        val_rows = []
        for alpha in args.weights:
            blended = (1 - alpha) * val_last + alpha * val_pred
            arrays = replay(pt, cap, val_true, blended, trigger,
                            args.report_threshold, "persistent_overflow",
                            penalty, max_moves=100)
            m = summarize(arrays)
            val_rows.append({"forecast_weight": alpha, "validation_metrics": m})
        feasible = [r for r in val_rows
                    if r["validation_metrics"]["route_changes_per_step"] <= args.budget]
        selection_pool = feasible or val_rows
        selected = min(selection_pool, key=lambda r: (
            r["validation_metrics"]["traffic_proxy_above_report_threshold_pct"],
            r["validation_metrics"]["steps_above_report_threshold_pct"],
            r["validation_metrics"]["mean_peak_utilization"],
            r["validation_metrics"]["route_changes_per_step"],
            r["validation_metrics"]["mean_igp_cost_stretch"],
            r["forecast_weight"],
        ))

        test_true, test_pred, test_last = test
        test_rows = {}
        test_arrays = {}
        for alpha in sorted(set(args.weights + [0.0, 1.0, selected["forecast_weight"]])):
            blended = (1 - alpha) * test_last + alpha * test_pred
            arrays = replay(pt, cap, test_true, blended, trigger,
                            args.report_threshold, "persistent_overflow",
                            penalty, max_moves=100)
            test_arrays[alpha] = arrays
            test_rows[str(alpha)] = summarize(arrays)

        baseline = test_arrays[0.0]
        chosen = test_arrays[selected["forecast_weight"]]
        full_forecast = test_arrays[1.0]
        scenario = {
            "capacity_scale": scale,
            "capacity_note": "Only scale 1.0 uses archived capacities; lower scales are synthetic.",
            "policy": control["policy"], "trigger": trigger,
            "reroute_penalty": penalty, "average_route_change_budget": args.budget,
            "validation_selection": selected,
            "validation_all_weights": val_rows,
            "test_by_forecast_weight": test_rows,
            "paired_test_differences": {
                "validation_selected_blend_minus_reactive": differences(
                    chosen, baseline, args.block, args.boot, 12000 + int(scale * 100)),
                "full_forecast_minus_reactive": differences(
                    full_forecast, baseline, args.block, args.boot, 13000 + int(scale * 100)),
            },
        }
        report["scenarios"][str(scale)] = scenario
        print(f"scale={scale:g} trigger={trigger:.0%} selected forecast weight="
              f"{selected['forecast_weight']:g}; test selected={test_rows[str(selected['forecast_weight'])]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Validation-tuned forecast blending for persistent routing", "",
          "Forecast weights were selected on validation. Test results are exploratory because this test split was inspected during development. "
          "At each capacity scale, all blend weights use the same validation-selected trigger, persistent optimizer, and penalty. "
          "The lower capacity scales are synthetic stress scenarios.", ""]
    for scale, scenario in report["scenarios"].items():
        md += [f"## Capacity scale {float(scale):g}", "",
               f"Trigger: {scenario['trigger']:.0%}; validation-selected weight: "
               f"{scenario['validation_selection']['forecast_weight']:g}; average route-change budget: {args.budget:g}/step.", "",
               "| Forecast weight | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | IGP stretch |",
               "|---:|---:|---:|---:|---:|---:|"]
        for alpha, metrics in sorted(scenario["test_by_forecast_weight"].items(), key=lambda x: float(x[0])):
            md.append(f"| {float(alpha):.2f} | {metrics['steps_above_report_threshold_pct']:.3f}% | "
                      f"{metrics['traffic_proxy_above_report_threshold_pct']:.6f}% | "
                      f"{metrics['mean_peak_utilization']:.3f} | {metrics['route_changes_per_step']:.3f} | "
                      f"{metrics['mean_igp_cost_stretch']:.4f} |")
        md += ["", "Paired proactive-minus-reactive differences (negative favors the blend):", "",
               "```json", json.dumps(scenario["paired_test_differences"], indent=2), "```", ""]
    mdpath = out.with_suffix(".md")
    mdpath.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {out} and {mdpath}")


if __name__ == "__main__":
    main()
