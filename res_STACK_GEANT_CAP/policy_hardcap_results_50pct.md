# Validation-selected routing policy comparison

Test period was inspected during development; these results are exploratory. All controllers use the same link capacities and trigger, and are limited to three final whole-flow route changes per step. The capacity scales below 1.0 are synthetic stress scenarios.

## Capacity scale 0.5

Validation selected persistent_overflow at a 55% trigger (penalty 0); hard route-change cap is 3 per step.

| Test policy | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | Alternate flows/step | IGP stretch |
|---|---:|---:|---:|---:|---:|---:|
| reactive_independent_baseline | 29.794% | 0.031232% | 0.613 | 0.046 | 12.91 | 1.0163 |
| proactive_independent | 29.107% | 0.031151% | 0.612 | 0.027 | 23.84 | 1.0133 |
| reactive_tuned_policy | 29.044% | 0.031062% | 0.611 | 0.042 | 16.96 | 1.0232 |
| proactive_tuned_policy | 29.294% | 0.031039% | 0.613 | 0.021 | 23.74 | 1.0149 |

Paired proactive-minus-reactive differences and 95% moving-block bootstrap intervals:

```json
{
  "reactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.749531542785759,
      "ci95_low": -1.9378513429106807,
      "ci95_high": 0.12492192379762648
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.00017011836659525003,
      "ci95_low": -0.0014569159965232207,
      "ci95_high": 0.0013995798971143333
    },
    "physical_overload_traffic_pp": {
      "mean": -2.1072219503328736e-05,
      "ci95_low": -4.1266604844031543e-05,
      "ci95_high": -7.77778246870034e-06
    },
    "mean_peak_utilization": {
      "mean": -0.0013475461649492635,
      "ci95_low": -0.003657834387752579,
      "ci95_high": 0.000508826076612108
    },
    "route_changes_per_step": {
      "mean": -0.003123048094940662,
      "ci95_low": -0.00938475952529669,
      "ci95_high": 0.0018738288569643974
    },
    "mean_igp_cost_stretch": {
      "mean": 0.0068718022645626806,
      "ci95_low": 0.004478553760223964,
      "ci95_high": 0.00993997356210293
    }
  },
  "proactive_tuned_policy_minus_reactive_tuned_policy": {
    "steps_above_report_threshold_pp": {
      "mean": 0.24984384759525297,
      "ci95_low": -0.18738288569643974,
      "ci95_high": 0.9369144284821986
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -2.318822689057146e-05,
      "ci95_low": -0.0024649391223722384,
      "ci95_high": 0.0015206097063094736
    },
    "physical_overload_traffic_pp": {
      "mean": 1.730515754335362e-05,
      "ci95_low": -1.6045015935069693e-05,
      "ci95_high": 4.815925933847453e-05
    },
    "mean_peak_utilization": {
      "mean": 0.0022314345553694184,
      "ci95_low": 0.0005515077844178356,
      "ci95_high": 0.004488921721118602
    },
    "route_changes_per_step": {
      "mean": -0.021236727045596503,
      "ci95_low": -0.06808244846970643,
      "ci95_high": -0.006230480949406635
    },
    "mean_igp_cost_stretch": {
      "mean": -0.008272358334127832,
      "ci95_low": -0.011291328401811325,
      "ci95_high": -0.006145940421864326
    }
  },
  "proactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.49968769519050593,
      "ci95_low": -1.9987507807620237,
      "ci95_high": 0.5621486570893192
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.0001933065934858215,
      "ci95_low": -0.0013308980035722233,
      "ci95_high": 0.00039595274741906304
    },
    "physical_overload_traffic_pp": {
      "mean": -3.767061959975112e-06,
      "ci95_low": -4.031093314578539e-05,
      "ci95_high": 2.2154521781952736e-05
    },
    "mean_peak_utilization": {
      "mean": 0.0008838883904201549,
      "ci95_low": -0.001715565541250716,
      "ci95_high": 0.0038019505452171554
    },
    "route_changes_per_step": {
      "mean": -0.024359775140537165,
      "ci95_low": -0.07059650218613367,
      "ci95_high": -0.008744534665833853
    },
    "mean_igp_cost_stretch": {
      "mean": -0.0014005560695651512,
      "ci95_low": -0.002431544992888034,
      "ci95_high": -0.0007685813922806055
    }
  },
  "proactive_independent_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.6870705808869456,
      "ci95_low": -1.8738288569643973,
      "ci95_high": 0.18738288569643974
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -8.054415267300593e-05,
      "ci95_low": -0.0013067721398116303,
      "ci95_high": 0.0006625250061411374
    },
    "physical_overload_traffic_pp": {
      "mean": 1.3537382558847012e-05,
      "ci95_low": -3.075018701356515e-05,
      "ci95_high": 6.542008356969121e-05
    },
    "mean_peak_utilization": {
      "mean": -0.00020397535940023753,
      "ci95_low": -0.0031169848278081095,
      "ci95_high": 0.0023343685993472162
    },
    "route_changes_per_step": {
      "mean": -0.018738288569643973,
      "ci95_low": -0.05624609618988132,
      "ci95_high": -0.006870705808869456
    },
    "mean_igp_cost_stretch": {
      "mean": -0.0029418429025871843,
      "ci95_low": -0.004570742453272011,
      "ci95_high": -0.0013262184473858112
    }
  }
}
```
