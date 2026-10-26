# Routing replay summary

Test replay on archived GÉANT per-link capacities and IGP weights. The test period has been inspected during development; treat these results as exploratory.

Configuration: **5 paths, overflow objective; reactive/oracle threshold 45%, proactive threshold 45%**. All routing controllers use the same threshold in this comparison.

No policy exceeded full link capacity in this replay. The table reports utilization above the chosen 45% controller threshold; that threshold is an operating choice, not physical congestion.

| Policy | Trigger | Steps >45% | Traffic proxy >45% | Mean peak util | P95 peak util | Flows rerouted / step | IGP cost stretch |
|---|---|---|---|---|---|---|---|
| static | 45% | 55.091% | 0.447% | 0.492 | 0.725 | 0.000 | 1.000 |
| reactive (last matrix) | 45% | 27.233% | 0.091% | 0.412 | 0.535 | 2.192 | 1.002 |
| proactive (same 0.45 trigger for all) | 45% | 28.170% | 0.094% | 0.416 | 0.519 | 2.212 | 1.002 |
| hybrid (max fcst, last) | 45% | 20.362% | 0.056% | 0.400 | 0.509 | 2.763 | 1.003 |
| oracle (true matrix) | 45% | 10.681% | 0.016% | 0.405 | 0.484 | 2.191 | 1.002 |

## Link-load estimation at a common threshold

| Input | Link-load RMSE (% capacity) | Precision | Recall | F1 |
|---|---|---|---|---|
| last observed (4 step(s) old) | 1.305 | 0.858 | 0.858 | 0.858 |
| same 0.45 trigger for all forecast | 1.020 | 0.890 | 0.862 | 0.876 |

The forecast's link-load RMSE is lower on the same targets and links. Detection scores use the same 45% threshold for truth and estimates; they are threshold-specific.

## Persistent route stability comparison

| Input | Action threshold | Handling | Steps >70% | Traffic proxy >70% | Route changes / step | Flows rerouted / step | IGP cost stretch |
|---|---|---|---|---|---|---|---|
| reactive | 70% | Independent | 5.434% | 0.004% | 0.105 | 0.081 | 1.000 |
| reactive | 70% | Persistent (h=0.4) | 0.375% | 0.001% | 0.002 | 2.012 | 1.001 |
| proactive | 45% | Independent | 0.062% | 0.000% | 0.941 | 2.212 | 1.002 |
| proactive | 45% | Persistent (h=0.01) | 0.000% | 0.000% | 0.005 | 7.279 | 1.009 |

Persistent-route rows use the separately validation-selected threshold for each input type; compare independent vs persistent within each input. Utilization results do not establish packet-loss, delay, or throughput improvement.