# Validation-selected routing policy comparison

Test period was inspected during development; these results are exploratory. All controllers use the same link capacities, trigger, and average route-change budget. The capacity scales below 1.0 are synthetic stress scenarios.

## Capacity scale 1

Validation selected persistent_overflow at a 55% trigger (penalty 0); average route-change budget 3/step met for both inputs: True.

| Test policy | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | Alternate flows/step | IGP stretch |
|---|---:|---:|---:|---:|---:|---:|
| reactive_independent_baseline | 0.437% | 0.001363% | 0.453 | 0.335 | 0.44 | 1.0002 |
| proactive_independent | 0.125% | 0.001080% | 0.451 | 0.176 | 0.33 | 1.0001 |
| reactive_tuned_policy | 0.000% | 0.000000% | 0.402 | 0.007 | 6.64 | 1.0012 |
| proactive_tuned_policy | 0.000% | 0.000000% | 0.383 | 0.004 | 3.15 | 1.0041 |

Paired proactive-minus-reactive differences and 95% moving-block bootstrap intervals:

```json
{
  "reactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.4372267332916927,
      "ci95_low": -0.8744534665833854,
      "ci95_high": -0.12492192379762648
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.0013634235733249768,
      "ci95_low": -0.0035786779962172052,
      "ci95_high": -5.115923308287465e-05
    },
    "physical_overload_traffic_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": -0.051006617553899763,
      "ci95_low": -0.060623438395722665,
      "ci95_high": -0.04189088871489645
    },
    "route_changes_per_step": {
      "mean": -0.3279200499687695,
      "ci95_low": -0.5146783260462211,
      "ci95_high": -0.18363522798251092
    },
    "mean_igp_cost_stretch": {
      "mean": 0.0010017735491498498,
      "ci95_low": 0.0007496897726851973,
      "ci95_high": 0.0012426301470088831
    }
  },
  "proactive_tuned_policy_minus_reactive_tuned_policy": {
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
      "mean": -0.018701433214003558,
      "ci95_low": -0.043150106266252625,
      "ci95_high": 0.0032993345705793387
    },
    "route_changes_per_step": {
      "mean": -0.003123048094940662,
      "ci95_low": -0.008744534665833853,
      "ci95_high": 0.0012492192379762648
    },
    "mean_igp_cost_stretch": {
      "mean": 0.002926549783768428,
      "ci95_low": 0.0006481935159421464,
      "ci95_high": 0.005212860361838168
    }
  },
  "proactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.4372267332916927,
      "ci95_low": -0.8744534665833854,
      "ci95_high": -0.12492192379762648
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.0013634235733249768,
      "ci95_low": -0.0035954695137333692,
      "ci95_high": -4.7433651927893496e-05
    },
    "physical_overload_traffic_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": -0.06970805076790333,
      "ci95_low": -0.09435048226004165,
      "ci95_high": -0.044489067364519
    },
    "route_changes_per_step": {
      "mean": -0.3310430980637102,
      "ci95_low": -0.5134603372891943,
      "ci95_high": -0.1792473454091194
    },
    "mean_igp_cost_stretch": {
      "mean": 0.003928323332918277,
      "ci95_low": 0.0014910352794011666,
      "ci95_high": 0.006423490841954052
    }
  },
  "proactive_independent_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.3123048094940662,
      "ci95_low": -0.6246096189881324,
      "ci95_high": -0.06246096189881324
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.0002829635476979714,
      "ci95_low": -0.0007675339099599152,
      "ci95_high": -1.72361518767692e-05
    },
    "physical_overload_traffic_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": -0.002015758927057565,
      "ci95_low": -0.0043214224483689835,
      "ci95_high": -0.00022608186260824497
    },
    "route_changes_per_step": {
      "mean": -0.15927545284197375,
      "ci95_low": -0.28235477826358524,
      "ci95_high": -0.0637101811367895
    },
    "mean_igp_cost_stretch": {
      "mean": -3.626451426088652e-05,
      "ci95_low": -8.136903414177239e-05,
      "ci95_high": -4.996782552633283e-06
    }
  }
}
```
