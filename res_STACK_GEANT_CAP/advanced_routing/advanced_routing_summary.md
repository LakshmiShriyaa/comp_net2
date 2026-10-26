# Advanced routing comparison

All methods use a shared action trigger, fixed reporting threshold, five IGP-shortest candidate paths, and validation-selected regularization. Scale 1.0 uses archived capacities; lower scales are synthetic stress cases. Test results are exploratory. MCF splits OD demand fractionally, so its split-flow equivalents are not whole-flow reroutes.

## Validation-selected parameters

| Capacity scale | Input | Reroute penalty | Change budget met | MCF path penalty (shared) | MCF split budget met |
|---:|---|---:|---|---:|---|
| 100% | reactive | 0 | yes | 0.01 | no |
| 100% | proactive | 0 | yes | 0.01 | no |
| 35% | reactive | 0 | yes | 0.01 | no |
| 35% | proactive | 0 | yes | 0.01 | no |

## Test replay

Traffic proxies estimate demand above a utilization level; they are not measured packet loss. Route changes count whole-flow path assignment changes. MCF uses fractional split-flow equivalents.

| Scale | Input | Policy | Steps >70% | Proxy >70% | Proxy >capacity | Mean peak | Changes/split-equivalents per step | OD flow on alternate paths per step | IGP stretch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 100% | reactive | Greedy overflow | 0.437% | 0.0014% | 0.0000% | 0.453 | 0.33 | 0.44 | 1.0002 |
| 100% | reactive | Persistent reroute-cost | 0.000% | 0.0000% | 0.0000% | 0.402 | 0.01 | 6.64 | 1.0012 |
| 100% | reactive | Fractional MCF | 0.062% | 0.0000% | 0.0000% | 0.413 | 9.42 | 17.51 | 1.0007 |
| 100% | proactive | Greedy overflow | 0.125% | 0.0011% | 0.0000% | 0.451 | 0.18 | 0.33 | 1.0001 |
| 100% | proactive | Persistent reroute-cost | 0.000% | 0.0000% | 0.0000% | 0.383 | 0.00 | 3.15 | 1.0041 |
| 100% | proactive | Fractional MCF | 0.000% | 0.0000% | 0.0000% | 0.416 | 6.21 | 17.80 | 1.0006 |
| 35% | reactive | Greedy overflow | 71.268% | 1.0795% | 0.1964% | 0.898 | 3.86 | 10.56 | 1.0096 |
| 35% | reactive | Persistent reroute-cost | 65.209% | 0.2987% | 0.0290% | 0.864 | 0.12 | 33.84 | 1.0251 |
| 35% | reactive | Fractional MCF | 51.968% | 1.1648% | 0.1522% | 0.780 | 26.01 | 70.93 | 1.0070 |
| 35% | proactive | Greedy overflow | 68.770% | 0.9717% | 0.1504% | 0.883 | 2.50 | 11.83 | 1.0097 |
| 35% | proactive | Persistent reroute-cost | 65.522% | 0.3331% | 0.0290% | 0.864 | 0.04 | 42.08 | 1.0250 |
| 35% | proactive | Fractional MCF | 48.532% | 1.0852% | 0.1215% | 0.760 | 19.15 | 78.39 | 1.0069 |

## Interpretation

- The persistent optimizer is warm-started from the previous routes. Validation selected a zero added reroute penalty; positive penalties worsened the chosen load metric. Its gains therefore come from retaining route state, not from a positive penalty value.
- On the synthetic 35% capacity case, persistence reduces the above-capacity traffic proxy substantially but keeps many flows on alternate paths and raises mean IGP stretch to about 2.5%.
- MCF is an idealized splittable-flow benchmark. It lowers peak utilization under stress, but no tested MCF setting met the three split-flow-equivalent budget.
- No block-bootstrap intervals were computed for these advanced methods. The test period was inspected during development, and reduced-capacity results are synthetic sensitivity tests.
