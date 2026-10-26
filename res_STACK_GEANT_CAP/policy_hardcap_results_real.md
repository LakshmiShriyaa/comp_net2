# Validation-selected routing policy comparison

Test period was inspected during development; these results are exploratory. All controllers use the same link capacities and trigger, and are limited to three final whole-flow route changes per step. The capacity scales below 1.0 are synthetic stress scenarios.

## Capacity scale 1

Validation selected independent_overflow at a 55% trigger (penalty 0); hard route-change cap is 3 per step.

| Test policy | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | Alternate flows/step | IGP stretch |
|---|---:|---:|---:|---:|---:|---:|
| reactive_independent_baseline | 0.062% | 0.000022% | 0.398 | 0.008 | 3.54 | 1.0034 |
| proactive_independent | 0.000% | 0.000000% | 0.373 | 0.004 | 4.03 | 1.0045 |
| reactive_tuned_policy | 0.062% | 0.000022% | 0.398 | 0.008 | 3.54 | 1.0034 |
| proactive_tuned_policy | 0.000% | 0.000000% | 0.373 | 0.004 | 4.03 | 1.0045 |

Paired proactive-minus-reactive differences and 95% moving-block bootstrap intervals:

```json
{
  "reactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "physical_overload_traffic_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "route_changes_per_step": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_igp_cost_stretch": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    }
  },
  "proactive_tuned_policy_minus_reactive_tuned_policy": {
    "steps_above_report_threshold_pp": {
      "mean": -0.06246096189881324,
      "ci95_low": -0.18738288569643974,
      "ci95_high": 0.0
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -2.1888185577521765e-05,
      "ci95_low": -6.56645567325653e-05,
      "ci95_high": 0.0
    },
    "physical_overload_traffic_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": -0.024784220458193916,
      "ci95_low": -0.04687254911824959,
      "ci95_high": -0.01242459253602889
    },
    "route_changes_per_step": {
      "mean": -0.0037476577139287947,
      "ci95_low": -0.010618363522798251,
      "ci95_high": 0.0018738288569643974
    },
    "mean_igp_cost_stretch": {
      "mean": 0.001175197299799726,
      "ci95_low": 0.00013139594504922846,
      "ci95_high": 0.0028057797152496104
    }
  },
  "proactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.06246096189881324,
      "ci95_low": -0.18738288569643974,
      "ci95_high": 0.0
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -2.1888185577521765e-05,
      "ci95_low": -6.56645567325653e-05,
      "ci95_high": 0.0
    },
    "physical_overload_traffic_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": -0.024784220458193916,
      "ci95_low": -0.04617740705697742,
      "ci95_high": -0.011158553469253665
    },
    "route_changes_per_step": {
      "mean": -0.0037476577139287947,
      "ci95_low": -0.010618363522798251,
      "ci95_high": 0.0018738288569643974
    },
    "mean_igp_cost_stretch": {
      "mean": 0.001175197299799726,
      "ci95_low": 0.00013620545702971033,
      "ci95_high": 0.0028203838176183564
    }
  },
  "proactive_independent_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.06246096189881324,
      "ci95_low": -0.18738288569643974,
      "ci95_high": 0.0
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -2.1888185577521765e-05,
      "ci95_low": -6.56645567325653e-05,
      "ci95_high": 0.0
    },
    "physical_overload_traffic_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": -0.024784220458193916,
      "ci95_low": -0.04702532792918798,
      "ci95_high": -0.010861279821019865
    },
    "route_changes_per_step": {
      "mean": -0.0037476577139287947,
      "ci95_low": -0.010633978763272954,
      "ci95_high": 0.0018738288569643974
    },
    "mean_igp_cost_stretch": {
      "mean": 0.001175197299799726,
      "ci95_low": 0.00014398165358700646,
      "ci95_high": 0.0028515419766285653
    }
  }
}
```
