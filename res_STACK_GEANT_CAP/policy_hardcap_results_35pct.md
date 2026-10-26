# Validation-selected routing policy comparison

Test period was inspected during development; these results are exploratory. All controllers use the same link capacities and trigger, and are limited to three final whole-flow route changes per step. The capacity scales below 1.0 are synthetic stress scenarios.

## Capacity scale 0.35

Validation selected persistent_overflow at a 45% trigger (penalty 0); hard route-change cap is 3 per step.

| Test policy | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | Alternate flows/step | IGP stretch |
|---|---:|---:|---:|---:|---:|---:|
| reactive_independent_baseline | 65.459% | 0.300803% | 0.864 | 0.157 | 32.85 | 1.0270 |
| proactive_independent | 65.584% | 0.302847% | 0.863 | 0.072 | 33.83 | 1.0252 |
| reactive_tuned_policy | 65.209% | 0.301924% | 0.863 | 0.092 | 35.96 | 1.0253 |
| proactive_tuned_policy | 65.522% | 0.328451% | 0.863 | 0.035 | 34.44 | 1.0249 |

Paired proactive-minus-reactive differences and 95% moving-block bootstrap intervals:

```json
{
  "reactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -0.24984384759525297,
      "ci95_low": -0.6870705808869456,
      "ci95_high": 0.0
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": 0.001120665782333091,
      "ci95_low": -0.013679052319696459,
      "ci95_high": 0.01459085151912479
    },
    "physical_overload_traffic_pp": {
      "mean": 0.00046701012138541253,
      "ci95_low": 0.00016044960093973217,
      "ci95_high": 0.0008323468440865634
    },
    "mean_peak_utilization": {
      "mean": -0.0007472657770160434,
      "ci95_low": -0.001151584816176321,
      "ci95_high": -0.0001441088683675186
    },
    "route_changes_per_step": {
      "mean": -0.06433479075577764,
      "ci95_low": -0.0924578388507183,
      "ci95_high": -0.021861336664584636
    },
    "mean_igp_cost_stretch": {
      "mean": -0.0016750286821482587,
      "ci95_low": -0.002392928858112451,
      "ci95_high": -0.001124052738039294
    }
  },
  "proactive_tuned_policy_minus_reactive_tuned_policy": {
    "steps_above_report_threshold_pp": {
      "mean": 0.3123048094940662,
      "ci95_low": 0.0,
      "ci95_high": 0.749531542785759
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": 0.02652686894060782,
      "ci95_low": 0.0016291556812994394,
      "ci95_high": 0.06435984103628485
    },
    "physical_overload_traffic_pp": {
      "mean": 1.8087959356231e-05,
      "ci95_low": -0.00011056755713221567,
      "ci95_high": 6.960929583454095e-05
    },
    "mean_peak_utilization": {
      "mean": 0.00017308001217434065,
      "ci95_low": -0.0009837737759525221,
      "ci95_high": 0.001287773536805582
    },
    "route_changes_per_step": {
      "mean": -0.05746408494690818,
      "ci95_low": -0.10869768894440975,
      "ci95_high": -0.021236727045596503
    },
    "mean_igp_cost_stretch": {
      "mean": -0.000375967288117335,
      "ci95_low": -0.0007857806071381142,
      "ci95_high": -6.646247390584348e-05
    }
  },
  "proactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": 0.06246096189881324,
      "ci95_low": -0.6246096189881324,
      "ci95_high": 0.6246096189881324
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": 0.02764753472294091,
      "ci95_low": 0.00979246669177004,
      "ci95_high": 0.05560261303758718
    },
    "physical_overload_traffic_pp": {
      "mean": 0.00048509808074164356,
      "ci95_low": 0.00015511484317693846,
      "ci95_high": 0.000814645331263797
    },
    "mean_peak_utilization": {
      "mean": -0.0005741857648417028,
      "ci95_low": -0.0016564481621195123,
      "ci95_high": 0.000575912504141067
    },
    "route_changes_per_step": {
      "mean": -0.12179887570268583,
      "ci95_low": -0.15677701436602123,
      "ci95_high": -0.08118363522798253
    },
    "mean_igp_cost_stretch": {
      "mean": -0.002050995970265594,
      "ci95_low": -0.002840079794625891,
      "ci95_high": -0.0015506452488620274
    }
  },
  "proactive_independent_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": 0.12492192379762648,
      "ci95_low": -0.12492192379762648,
      "ci95_high": 0.4372267332916927
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": 0.0020438920591431342,
      "ci95_low": 0.0015579019795807004,
      "ci95_high": 0.007161385086117253
    },
    "physical_overload_traffic_pp": {
      "mean": 0.00024136888443421258,
      "ci95_low": -7.060227215493924e-05,
      "ci95_high": 0.0004451932470133649
    },
    "mean_peak_utilization": {
      "mean": -0.0007668608691378134,
      "ci95_low": -0.0019256549345784302,
      "ci95_high": 1.2470781337760573e-05
    },
    "route_changes_per_step": {
      "mean": -0.08432229856339787,
      "ci95_low": -0.11683322923173016,
      "ci95_high": -0.05495003123048096
    },
    "mean_igp_cost_stretch": {
      "mean": -0.0017571884435319354,
      "ci95_low": -0.00211895726712048,
      "ci95_high": -0.0014005103395540255
    }
  }
}
```
