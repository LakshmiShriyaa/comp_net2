# Validation-tuned forecast blending for persistent routing

Forecast weights were selected on validation. Test results are exploratory because this test split was inspected during development. At each capacity scale, all blend weights use the same validation-selected trigger, persistent optimizer, and penalty. The lower capacity scales are synthetic stress scenarios.

## Capacity scale 1

Trigger: 55%; validation-selected weight: 0.5; average route-change budget: 3/step.

| Forecast weight | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | IGP stretch |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.000% | 0.000000% | 0.402 | 0.007 | 1.0012 |
| 0.10 | 0.000% | 0.000000% | 0.403 | 0.006 | 1.0011 |
| 0.25 | 0.000% | 0.000000% | 0.406 | 0.006 | 1.0010 |
| 0.50 | 0.000% | 0.000000% | 0.407 | 0.004 | 1.0010 |
| 0.75 | 0.000% | 0.000000% | 0.373 | 0.004 | 1.0045 |
| 1.00 | 0.000% | 0.000000% | 0.383 | 0.004 | 1.0041 |

Paired proactive-minus-reactive differences (negative favors the blend):

```json
{
  "validation_selected_blend_minus_reactive": {
    "steps_above_70_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "traffic_proxy_above_70_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "physical_overload_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": 0.005035220701794264,
      "ci95_low": 0.0004012911776540281,
      "ci95_high": 0.009172038689749005
    },
    "route_changes_per_step": {
      "mean": -0.0024984384759525295,
      "ci95_low": -0.008119925046845722,
      "ci95_high": 0.0018738288569643974
    },
    "mean_igp_cost_stretch": {
      "mean": -0.00018015464214873056,
      "ci95_low": -0.00032354204165408726,
      "ci95_high": -1.6386679658422098e-05
    }
  },
  "full_forecast_minus_reactive": {
    "steps_above_70_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "traffic_proxy_above_70_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "physical_overload_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "mean_peak_utilization": {
      "mean": -0.018701433214003558,
      "ci95_low": -0.04327699950012581,
      "ci95_high": 0.004297373394745353
    },
    "route_changes_per_step": {
      "mean": -0.003123048094940662,
      "ci95_low": -0.009369144284821987,
      "ci95_high": 0.0012492192379762648
    },
    "mean_igp_cost_stretch": {
      "mean": 0.002926549783768428,
      "ci95_low": 0.000714938152641271,
      "ci95_high": 0.005144281441046388
    }
  }
}
```

## Capacity scale 0.5

Trigger: 55%; validation-selected weight: 0.25; average route-change budget: 3/step.

| Forecast weight | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | IGP stretch |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 29.044% | 0.031068% | 0.611 | 0.058 | 1.0232 |
| 0.10 | 29.044% | 0.030048% | 0.611 | 0.044 | 1.0229 |
| 0.25 | 28.982% | 0.030005% | 0.610 | 0.074 | 1.0229 |
| 0.50 | 29.044% | 0.030018% | 0.612 | 0.039 | 1.0212 |
| 0.75 | 29.044% | 0.030010% | 0.612 | 0.031 | 1.0211 |
| 1.00 | 29.294% | 0.030946% | 0.613 | 0.021 | 1.0149 |

Paired proactive-minus-reactive differences (negative favors the blend):

```json
{
  "validation_selected_blend_minus_reactive": {
    "steps_above_70_pp": {
      "mean": -0.06246096189881324,
      "ci95_low": -0.3123048094940662,
      "ci95_high": 0.12492192379762648
    },
    "traffic_proxy_above_70_pp": {
      "mean": -0.001063085937985795,
      "ci95_low": -0.0031764820587666318,
      "ci95_high": 0.0002101195685436821
    },
    "physical_overload_pp": {
      "mean": -4.670240097487014e-06,
      "ci95_low": -2.9161763997449466e-05,
      "ci95_high": 1.063624158274702e-05
    },
    "mean_peak_utilization": {
      "mean": -0.0008886239626942241,
      "ci95_low": -0.0016129987522596721,
      "ci95_high": -0.00019642125296265094
    },
    "route_changes_per_step": {
      "mean": 0.01561524047470331,
      "ci95_low": -0.028747657713928794,
      "ci95_high": 0.05871330418488445
    },
    "mean_igp_cost_stretch": {
      "mean": -0.0002552071421295532,
      "ci95_low": -0.00032734056268811384,
      "ci95_high": -0.0001856350845696245
    }
  },
  "full_forecast_minus_reactive": {
    "steps_above_70_pp": {
      "mean": 0.24984384759525297,
      "ci95_low": -0.18738288569643974,
      "ci95_high": 0.9369144284821986
    },
    "traffic_proxy_above_70_pp": {
      "mean": -0.00012276246906867784,
      "ci95_low": -0.002519109847973344,
      "ci95_high": 0.0014755751351217005
    },
    "physical_overload_pp": {
      "mean": 1.7332824815283283e-05,
      "ci95_low": -1.3207688116645335e-05,
      "ci95_high": 4.6188816659556886e-05
    },
    "mean_peak_utilization": {
      "mean": 0.0021026229798831023,
      "ci95_low": 0.0004411844965966324,
      "ci95_high": 0.00458290471947245
    },
    "route_changes_per_step": {
      "mean": -0.03685196752029981,
      "ci95_low": -0.10243597751405371,
      "ci95_high": -0.007495315427857589
    },
    "mean_igp_cost_stretch": {
      "mean": -0.008272788551771675,
      "ci95_low": -0.011266601424072882,
      "ci95_high": -0.006126753775869462
    }
  }
}
```

## Capacity scale 0.35

Trigger: 45%; validation-selected weight: 0.1; average route-change budget: 3/step.

| Forecast weight | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | IGP stretch |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 65.209% | 0.295664% | 0.863 | 0.138 | 1.0259 |
| 0.10 | 65.209% | 0.300740% | 0.863 | 0.104 | 1.0255 |
| 0.25 | 65.272% | 0.324536% | 0.863 | 0.105 | 1.0268 |
| 0.50 | 65.397% | 0.324536% | 0.863 | 0.066 | 1.0268 |
| 0.75 | 65.459% | 0.327820% | 0.863 | 0.051 | 1.0249 |
| 1.00 | 65.522% | 0.327441% | 0.863 | 0.036 | 1.0251 |

Paired proactive-minus-reactive differences (negative favors the blend):

```json
{
  "validation_selected_blend_minus_reactive": {
    "steps_above_70_pp": {
      "mean": 0.0,
      "ci95_low": 0.0,
      "ci95_high": 0.0
    },
    "traffic_proxy_above_70_pp": {
      "mean": 0.00507620414073375,
      "ci95_low": -0.0008498130792335397,
      "ci95_high": 0.014836046604138757
    },
    "physical_overload_pp": {
      "mean": 3.048110684380717e-05,
      "ci95_low": -7.314247356146658e-05,
      "ci95_high": 0.00016302876729153634
    },
    "mean_peak_utilization": {
      "mean": -0.00017211440576008898,
      "ci95_low": -0.00042273817138326384,
      "ci95_high": 7.616309245476777e-06
    },
    "route_changes_per_step": {
      "mean": -0.034353529044347283,
      "ci95_low": -0.07495315427857589,
      "ci95_high": -0.006246096189881324
    },
    "mean_igp_cost_stretch": {
      "mean": -0.00040161248607667547,
      "ci95_low": -0.0009146717177315811,
      "ci95_high": -1.3329435752912137e-05
    }
  },
  "full_forecast_minus_reactive": {
    "steps_above_70_pp": {
      "mean": 0.3123048094940662,
      "ci95_low": 0.0,
      "ci95_high": 0.8119925046845722
    },
    "traffic_proxy_above_70_pp": {
      "mean": 0.03177683036339635,
      "ci95_low": 0.0075624506017370596,
      "ci95_high": 0.06690047994380303
    },
    "physical_overload_pp": {
      "mean": 4.33067897808282e-05,
      "ci95_low": -6.779940081952002e-05,
      "ci95_high": 0.00017430162852971373
    },
    "mean_peak_utilization": {
      "mean": 0.00014291265322073483,
      "ci95_low": -0.0008178255907600184,
      "ci95_high": 0.001305737627959862
    },
    "route_changes_per_step": {
      "mean": -0.10243597751405371,
      "ci95_low": -0.2180043722673329,
      "ci95_high": -0.024984384759525295
    },
    "mean_igp_cost_stretch": {
      "mean": -0.0007785957819075351,
      "ci95_low": -0.0012444682007204287,
      "ci95_high": -0.00044644963038990366
    }
  }
}
```
