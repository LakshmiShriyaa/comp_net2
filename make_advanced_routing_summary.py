"""Build a compact Markdown table from the advanced routing experiment JSON."""
import json
from pathlib import Path


SOURCE = Path("res_STACK_GEANT_CAP/advanced_routing/advanced_routing_results.json")
OUTPUT = SOURCE.with_name("advanced_routing_summary.md")


def main():
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    lines = [
        "# Advanced routing comparison", "",
        "All methods use a shared action trigger, fixed reporting threshold, five IGP-shortest candidate paths, "
        "and validation-selected regularization. Scale 1.0 uses archived capacities; lower scales are synthetic "
        "stress cases. Test results are exploratory. MCF splits OD demand fractionally, so its split-flow "
        "equivalents are not whole-flow reroutes.", "",
        "## Validation-selected parameters", "",
        "| Capacity scale | Input | Reroute penalty | Change budget met | MCF path penalty (shared) | MCF split budget met |",
        "|---:|---|---:|---|---:|---|",
    ]
    for scale, scenario in report["scenarios"].items():
        mcf = scenario["validation_tuning"]["mcf_joint"]
        for kind in ("reactive", "proactive"):
            result = scenario["validation_tuning"][f"{kind}_reroute_cost"]
            lines.append(
                f"| {float(scale):.0%} | {kind} | {result['selected']['penalty']:g} | "
                f"{'yes' if result['budget_met'] else 'no'} | {mcf['selected_penalty']:g} | "
                f"{'yes' if mcf['budget_met'] else 'no'} |"
            )
    lines += [
        "", "## Test replay", "",
        "Traffic proxies estimate demand above a utilization level; they are not measured packet loss. "
        "Route changes count whole-flow path assignment changes. MCF uses fractional split-flow equivalents.", "",
        "| Scale | Input | Policy | Steps >70% | Proxy >70% | Proxy >capacity | Mean peak | Changes/split-equivalents per step | OD flow on alternate paths per step | IGP stretch |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    methods = (("Greedy overflow", "greedy_overflow"),
               ("Persistent reroute-cost", "reroute_cost"),
               ("Fractional MCF", "mcf_split"))
    for scale, scenario in report["scenarios"].items():
        for kind in ("reactive", "proactive"):
            for label, suffix in methods:
                result = scenario["test"][f"{kind}_{suffix}"]
                lines.append(
                    f"| {float(scale):.0%} | {kind} | {label} | "
                    f"{result['steps_above_report_threshold_pct']:.3f}% | "
                    f"{result['above_threshold_traffic_proxy_pct']:.4f}% | "
                    f"{result['above_capacity_traffic_proxy_pct']:.4f}% | "
                    f"{result['mean_peak_utilization']:.3f} | "
                    f"{result['route_changes_or_split_equivalents_per_step']:.2f} | "
                    f"{result['flows_using_nonshortest_paths_per_step']:.2f} | "
                    f"{result['mean_igp_cost_stretch']:.4f} |"
                )
    lines += [
        "", "## Interpretation", "",
        "- The persistent optimizer is warm-started from the previous routes. Validation selected a zero added "
        "reroute penalty; positive penalties worsened the chosen load metric. Its gains therefore come from "
        "retaining route state, not from a positive penalty value.",
        "- On the synthetic 35% capacity case, persistence reduces the above-capacity traffic proxy substantially "
        "but keeps many flows on alternate paths and raises mean IGP stretch to about 2.5%.",
        "- MCF is an idealized splittable-flow benchmark. It lowers peak utilization under stress, but no tested "
        "MCF setting met the three split-flow-equivalent budget.",
        "- No block-bootstrap intervals were computed for these advanced methods. The test period was inspected "
        "during development, and reduced-capacity results are synthetic sensitivity tests.", "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
