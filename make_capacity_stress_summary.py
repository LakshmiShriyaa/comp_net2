"""Build the stress-scenario Markdown table from fair-evaluator JSON outputs."""
import json
from pathlib import Path


ROOT = Path("res_STACK_GEANT_CAP")
SCENARIOS = (("50% capacity", 0.50, "stress_50pct"),
             ("35% capacity", 0.35, "stress_35pct"))


def main():
    lines = [
        "# Capacity-scale sensitivity replay", "",
        "Capacity scales are synthetic stress scenarios applied uniformly to archived link capacities, not observed GÉANT capacities. "
        "All policies use the same 45%, 55%, 70% action-trigger grid, a fixed 70% reporting threshold, and a 3-flow/step validation budget.", "",
        "All test results are exploratory. Above-capacity traffic is a simulation proxy, not measured packet loss.", "",
    ]
    for title, scale, folder in SCENARIOS:
        path = ROOT / folder / "fair_routing_comparison.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        lines += [
            f"## {title} (scale {scale:.0%})", "",
            "| Common trigger | Policy | Steps >70% | Traffic proxy >70% | Proxy >capacity | Mean peak utilization | Reroutes/step |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for trigger in (0.45, 0.55, 0.70):
            for scheme, label in (("reactive", "Reactive"), ("proactive", "Proactive")):
                m = report["test_grid"][f"{scheme}@{trigger}"]
                lines.append(
                    f"| {trigger:.0%} | {label} | {m['steps_above_%']:.3f}% | "
                    f"{m['above_traffic_%']:.4f}% | {m['physical_overload_%']:.4f}% | "
                    f"{m['mean_peak_util']:.3f} | {m['reroutes_per_step']:.2f} |"
                )
        lines += ["", "| Trigger | Steps difference pp [95% CI] | >70% proxy difference pp [95% CI] | Reroutes difference/step [95% CI] |",
                  "|---:|---:|---:|---:|"]
        for row in report["same_trigger_diffs"]:
            def fmt(values):
                return f"{values[0]:+.3f} [{values[1]:+.3f}, {values[2]:+.3f}]"
            lines.append(
                f"| {row['reactive_trigger']:.0%} | {fmt(row['d_steps_above_pp'])} | "
                f"{fmt(row['d_above_traffic_pp'])} | {fmt(row['d_reroutes'])} |"
            )
        lines.append("")
    output = ROOT / "capacity_stress_comparison.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
