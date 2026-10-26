# Validation-selected routing policy comparison

Test period was inspected during development; these results are exploratory. All controllers use the same link capacities, trigger, and average route-change budget. The capacity scales below 1.0 are synthetic stress scenarios.

## Capacity scale 0.5

Validation selected persistent_overflow at a 55% trigger (penalty 0); average route-change budget 3/step met for both inputs: True.

| Test policy | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | Alternate flows/step | IGP stretch |
|---|---:|---:|---:|---:|---:|---:|
| reactive_independent_baseline | 34.354% | 0.202245% | 0.653 | 2.488 | 5.40 | 1.0090 |
| proactive_independent | 31.043% | 0.157133% | 0.646 | 1.520 | 5.39 | 1.0092 |
| reactive_tuned_policy | 29.044% | 0.031068% | 0.611 | 0.058 | 17.01 | 1.0232 |
| proactive_tuned_policy | 29.294% | 0.030946% | 0.613 | 0.021 | 23.76 | 1.0149 |

Paired proactive-minus-reactive differences and 95% moving-block bootstrap intervals:

```json
{
  "reactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -5.309181761399126,
      "ci95_low": -6.997189256714553,
      "ci95_high": -3.7460961898813325
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.17117672405409814,
      "ci95_low": -0.24247410108219183,
      "ci95_high": -0.11807960715939743
    },
    "physical_overload_traffic_pp": {
      "mean": -0.009467010635809826,
      "ci95_low": -0.018610314133055406,
      "ci95_high": -0.002717838616344984
    },
    "mean_peak_utilization": {
      "mean": -0.04221680564928025,
      "ci95_low": -0.04842919255377053,
      "ci95_high": -0.0355140443921536
    },
    "route_changes_per_step": {
      "mean": -2.429731417863835,
      "ci95_low": -2.8032479700187385,
      "ci95_high": -2.11606808244847
    },
    "mean_igp_cost_stretch": {
      "mean": 0.014178352446254629,
      "ci95_low": 0.012471560007689963,
      "ci95_high": 0.01645676532643165
    }
  },
  "proactive_tuned_policy_minus_reactive_tuned_policy": {
    "steps_above_report_threshold_pp": {
      "mean": 0.24984384759525297,
      "ci95_low": -0.18738288569643974,
      "ci95_high": 0.9369144284821986
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.00012276246906867784,
      "ci95_low": -0.0024728383854205273,
      "ci95_high": 0.0015209250471844738
    },
    "physical_overload_traffic_pp": {
      "mean": 1.7332824815283283e-05,
      "ci95_low": -1.439257751409831e-05,
      "ci95_high": 4.584595323327304e-05
    },
    "mean_peak_utilization": {
      "mean": 0.0021026229798831023,
      "ci95_low": 0.00047504173581262144,
      "ci95_high": 0.004489601542155195
    },
    "route_changes_per_step": {
      "mean": -0.03685196752029981,
      "ci95_low": -0.1061836352279825,
      "ci95_high": -0.008104309806371104
    },
    "mean_igp_cost_stretch": {
      "mean": -0.008272788551771675,
      "ci95_low": -0.011334514963535778,
      "ci95_high": -0.006080565576919239
    }
  },
  "proactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -5.059337913803873,
      "ci95_low": -6.620861961274204,
      "ci95_high": -3.622735790131168
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.17129948652316684,
      "ci95_low": -0.23927725639501324,
      "ci95_high": -0.11966719347477957
    },
    "physical_overload_traffic_pp": {
      "mean": -0.009449677810994544,
      "ci95_low": -0.01833296404936153,
      "ci95_high": -0.0028524720128352047
    },
    "mean_peak_utilization": {
      "mean": -0.04011418266939715,
      "ci95_low": -0.04544056027960137,
      "ci95_high": -0.03333067719468023
    },
    "route_changes_per_step": {
      "mean": -2.466583385384135,
      "ci95_low": -2.8557307932542164,
      "ci95_high": -2.1617270455965025
    },
    "mean_igp_cost_stretch": {
      "mean": 0.005905563894482956,
      "ci95_low": 0.0031184323112949186,
      "ci95_high": 0.008846408575504285
    }
  },
  "proactive_independent_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -3.310430980637102,
      "ci95_low": -4.62211118051218,
      "ci95_high": -1.8738288569643973
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.04511238934536721,
      "ci95_low": -0.07572029935109927,
      "ci95_high": -0.01889298867850162
    },
    "physical_overload_traffic_pp": {
      "mean": -0.00032903134030870426,
      "ci95_low": -0.006416148842587878,
      "ci95_high": 0.008838666896602744
    },
    "mean_peak_utilization": {
      "mean": -0.007047793721497469,
      "ci95_low": -0.010528922654330024,
      "ci95_high": -0.0027907615775320423
    },
    "route_changes_per_step": {
      "mean": -0.9675202998126171,
      "ci95_low": -1.328575890068707,
      "ci95_high": -0.7588850718301062
    },
    "mean_igp_cost_stretch": {
      "mean": 0.00019242458830693516,
      "ci95_low": -3.0421278927776688e-05,
      "ci95_high": 0.0004035965281347598
    }
  }
}
```

## Capacity scale 0.35

Validation selected persistent_overflow at a 45% trigger (penalty 0); average route-change budget 3/step met for both inputs: True.

| Test policy | Steps >70% | >70% traffic proxy | Mean peak utilization | Route changes/step | Alternate flows/step | IGP stretch |
|---|---:|---:|---:|---:|---:|---:|
| reactive_independent_baseline | 74.391% | 1.233667% | 0.925 | 1.026 | 2.99 | 1.0091 |
| proactive_independent | 73.017% | 1.114732% | 0.916 | 0.711 | 3.00 | 1.0092 |
| reactive_tuned_policy | 65.209% | 0.295664% | 0.863 | 0.138 | 37.39 | 1.0259 |
| proactive_tuned_policy | 65.522% | 0.327441% | 0.863 | 0.036 | 35.14 | 1.0251 |

Paired proactive-minus-reactive differences and 95% moving-block bootstrap intervals:

```json
{
  "reactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -9.181761399125547,
      "ci95_low": -13.179262960649593,
      "ci95_high": -5.996252342286072
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.938002856075092,
      "ci95_low": -1.1878756659098995,
      "ci95_high": -0.7168284691601629
    },
    "physical_overload_traffic_pp": {
      "mean": -0.23281868758978078,
      "ci95_low": -0.31744851724507833,
      "ci95_high": -0.1629487872172926
    },
    "mean_peak_utilization": {
      "mean": -0.06237513020513655,
      "ci95_low": -0.07167954181439054,
      "ci95_high": -0.05600146344728503
    },
    "route_changes_per_step": {
      "mean": -0.8881948782011243,
      "ci95_low": -1.0312460961898813,
      "ci95_high": -0.725749531542786
    },
    "mean_igp_cost_stretch": {
      "mean": 0.01681968432926125,
      "ci95_low": 0.014742485452118892,
      "ci95_high": 0.01960698183890657
    }
  },
  "proactive_tuned_policy_minus_reactive_tuned_policy": {
    "steps_above_report_threshold_pp": {
      "mean": 0.3123048094940662,
      "ci95_low": 0.0,
      "ci95_high": 0.8119925046845722
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": 0.03177683036339635,
      "ci95_low": 0.006918343698801673,
      "ci95_high": 0.06867376121239635
    },
    "physical_overload_traffic_pp": {
      "mean": 4.33067897808282e-05,
      "ci95_low": -7.308444462570634e-05,
      "ci95_high": 0.00017459715489594332
    },
    "mean_peak_utilization": {
      "mean": 0.00014291265322073483,
      "ci95_low": -0.0007918624897289097,
      "ci95_high": 0.001320435552579175
    },
    "route_changes_per_step": {
      "mean": -0.10243597751405371,
      "ci95_low": -0.21486570893191756,
      "ci95_high": -0.026842598376015075
    },
    "mean_igp_cost_stretch": {
      "mean": -0.0007785957819075351,
      "ci95_low": -0.0012463042459242076,
      "ci95_high": -0.00041593753565325635
    }
  },
  "proactive_tuned_policy_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -8.869456589631481,
      "ci95_low": -13.179262960649593,
      "ci95_high": -5.8088694565896315
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.9062260257116956,
      "ci95_low": -1.141295791932028,
      "ci95_high": -0.7018505433148461
    },
    "physical_overload_traffic_pp": {
      "mean": -0.23277538079999993,
      "ci95_low": -0.3182673696205863,
      "ci95_high": -0.16622801909496243
    },
    "mean_peak_utilization": {
      "mean": -0.06223221755191581,
      "ci95_low": -0.07160856921242073,
      "ci95_high": -0.05569364924884305
    },
    "route_changes_per_step": {
      "mean": -0.9906308557151781,
      "ci95_low": -1.1037008119925047,
      "ci95_high": -0.8875702685821362
    },
    "mean_igp_cost_stretch": {
      "mean": 0.01604108854735372,
      "ci95_low": 0.014095831240942693,
      "ci95_high": 0.01877685211312734
    }
  },
  "proactive_independent_minus_reactive_independent_baseline": {
    "steps_above_report_threshold_pp": {
      "mean": -1.3741411617738912,
      "ci95_low": -2.373516552154903,
      "ci95_high": -0.8119925046845722
    },
    "traffic_proxy_above_report_threshold_pp": {
      "mean": -0.11893486178214949,
      "ci95_low": -0.18719760197881322,
      "ci95_high": -0.050011521151314825
    },
    "physical_overload_traffic_pp": {
      "mean": -0.06436308996443615,
      "ci95_low": -0.09788900158721303,
      "ci95_high": -0.03667466020395184
    },
    "mean_peak_utilization": {
      "mean": -0.009913230039118827,
      "ci95_low": -0.013091771959383587,
      "ci95_high": -0.006439514159411657
    },
    "route_changes_per_step": {
      "mean": -0.3154278575890069,
      "ci95_low": -0.4072610868207371,
      "ci95_high": -0.22609306683322933
    },
    "mean_igp_cost_stretch": {
      "mean": 0.00014142205906327444,
      "ci95_low": -4.475752439089543e-06,
      "ci95_high": 0.0003080411856248918
    }
  }
}
```
