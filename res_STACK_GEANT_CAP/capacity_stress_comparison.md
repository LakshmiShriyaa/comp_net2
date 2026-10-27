# Capacity-scale sensitivity replay

Capacity scales are synthetic stress scenarios applied uniformly to archived link capacities, not observed GÉANT capacities. All policies use the same 45%, 55%, 70% action-trigger grid, a fixed 70% reporting threshold, and a validation budget of 3 average flows assigned to non-default paths per step.

All test results are exploratory. Above-capacity traffic is a simulation proxy, not measured packet loss.

## 50% capacity (scale 50%)

| Common trigger | Policy | Steps >70% | Traffic proxy >70% | Proxy >capacity | Mean peak utilization | Flows off default/step |
|---:|---|---:|---:|---:|---:|---:|
| 45% | Reactive | 32.480% | 0.1973% | 0.0101% | 0.635 | 9.51 |
| 45% | Proactive | 30.106% | 0.1504% | 0.0089% | 0.625 | 10.57 |
| 55% | Reactive | 34.166% | 0.2002% | 0.0101% | 0.653 | 7.28 |
| 55% | Proactive | 30.918% | 0.1526% | 0.0089% | 0.645 | 7.93 |
| 70% | Reactive | 44.347% | 0.2523% | 0.0134% | 0.715 | 4.63 |
| 70% | Proactive | 44.285% | 0.1944% | 0.0105% | 0.709 | 4.89 |

| Trigger | Steps difference pp [95% CI] | >70% proxy difference pp [95% CI] | Off-default-flow difference/step [95% CI] |
|---:|---:|---:|---:|
| 45% | -2.374 [-3.123, -1.499] | -0.047 [-0.077, -0.018] | +1.063 [+0.750, +1.360] |
| 55% | -3.248 [-4.560, -1.874] | -0.048 [-0.078, -0.019] | +0.650 [+0.317, +0.969] |
| 70% | -0.062 [-1.936, +1.437] | -0.058 [-0.094, -0.025] | +0.258 [+0.025, +0.555] |

## 35% capacity (scale 35%)

| Common trigger | Policy | Steps >70% | Traffic proxy >70% | Proxy >capacity | Mean peak utilization | Flows off default/step |
|---:|---|---:|---:|---:|---:|---:|
| 45% | Reactive | 70.456% | 1.0857% | 0.1993% | 0.893 | 11.07 |
| 45% | Proactive | 68.520% | 0.9775% | 0.1531% | 0.880 | 12.52 |
| 55% | Reactive | 71.268% | 1.0795% | 0.1964% | 0.898 | 10.56 |
| 55% | Proactive | 68.770% | 0.9717% | 0.1504% | 0.883 | 11.83 |
| 70% | Reactive | 75.578% | 1.0883% | 0.1971% | 0.915 | 8.56 |
| 70% | Proactive | 75.453% | 0.9857% | 0.1511% | 0.902 | 9.49 |

| Trigger | Steps difference pp [95% CI] | >70% proxy difference pp [95% CI] | Off-default-flow difference/step [95% CI] |
|---:|---:|---:|---:|
| 45% | -1.936 [-2.873, -1.124] | -0.108 [-0.171, -0.043] | +1.450 [+1.057, +1.771] |
| 55% | -2.498 [-3.373, -1.624] | -0.108 [-0.171, -0.043] | +1.270 [+0.923, +1.556] |
| 70% | -0.125 [-1.437, +1.312] | -0.103 [-0.165, -0.037] | +0.931 [+0.592, +1.254] |
