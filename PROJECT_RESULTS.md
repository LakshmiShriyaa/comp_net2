# Project results and presentation guide

## What the project does

The project forecasts GÉANT origin–destination traffic matrices, converts those forecasts into link-load estimates, and compares static routing, reactive rerouting, and proactive rerouting. The traffic data comes from the public GÉANT matrix dataset, sampled every 15 minutes; the project uses 23 nodes and 10,772 recorded matrices. The dataset paper describes the public GÉANT matrices as 15-minute measurements.[^geant]

## Forecast experiment

- **Split:** chronological 70% train, 15% validation, 15% test. Missing time intervals are excluded from windows that cross a gap. The retained counts are 7,424 train, 1,601 validation, and 1,601 test windows.
- **Task:** predict four steps ahead, which is one hour on GÉANT. The table below reports the farthest forecast only, on the same held-out test targets, with diagonal OD entries excluded.
- **Candidate selection:** stack weights, ridge penalty, and residual correction gain are selected using expanding chronological folds within validation. The stack combines CNN-LSTM seed ensemble, normalized-loss CNN, per-OD ridge, seasonal ridge, log ridge, flow-tree ensemble, 3-step moving average, and shared LSTM forecasts. Some base-model checkpoint/hyperparameter choices used the full validation split, so these folds are not a fully nested independent estimate.
- **Metric:** normalized RMSE scales each off-diagonal OD pair by its range on the training prefix, then pools errors equally across OD pairs and forecast windows. Raw MAE/RMSE are also reported in kbps.

| Model | Test MAE (kbps) | Test RMSE (kbps) | MAPE | Per-OD normalized RMSE |
|---|---:|---:|---:|---:|
| Last-value baseline | 2,183.3 | 10,573.5 | 22.79% | 0.06893 |
| CNN-LSTM | 2,162.1 | 10,147.2 | 21.86% | 0.06372 |
| Per-OD ridge | 2,283.2 | 9,920.7 | 22.42% | 0.05772 |
| Flow-by-flow tree | 2,084.7 | 9,866.6 | 21.57% | 0.06242 |
| **Validation-CV-selected stack** | **1,995.1** | **9,692.4** | **21.02%** | **0.05579** |

Compared with last-value persistence, this stack reduces test MAE by **8.6%**, RMSE by **8.3%**, MAPE by **7.8%**, and per-OD normalized RMSE by **19.1%**. The mean chronological validation-fold improvement is **18.0%**, with the caveat above about base-model selection. This is close to the 20% target but does not reach it. Test results were inspected during model development, so treat this as exploratory rather than an untouched final benchmark; a new chronological test period is needed for independent confirmation.

## Routing experiment

### Capacity provenance and limits

The link table `geant_link_capacities.csv` was mapped from IGen's archived [GÉANT GML dated 25-Nov-2004](https://igen.sourceforge.net/downloads/geant-20041125.gml.html), not read from the traffic-matrix dataset itself. The GML contains link bandwidths in bps and IGP metrics; bandwidths were converted to kbps. The 23-node, 37-link graph has a unique graph-isomorphism match to the anonymized traffic topology after applying the node order in `dataset_info.txt`. The resulting capacity range is 155,000 kbps to 10,000,000 kbps. This is a plausible archived GÉANT topology mapping, but not proof that these are the exact capacities concurrent with the anonymized traffic measurements.

Under these assumed heterogeneous capacities, **none of the 1,601 test matrices overloaded a link**, including under static shortest-path routing. Static mean and p95 peak-link utilization were 0.492 and 0.725. Therefore this experiment does not demonstrate congestion avoidance or reduced packet loss. Utilization above 45% below is an operating threshold selected for the controller, not physical congestion.

### Matched-threshold result

The table below uses the same 45% action threshold for reactive, proactive, hybrid, and oracle routing, with the same topology, capacities, five candidate paths, and overflow objective. This isolates forecast input while holding the controller setting fixed. Static routing is shown as the no-rerouting reference.

| Policy | Steps above 45% | Traffic proxy above 45% | Mean peak utilization | p95 peak utilization | Flows rerouted / interval | Mean IGP-cost stretch |
|---|---:|---:|---:|---:|---:|---:|
| Static shortest path | 55.091% | 0.446664% | 0.492 | 0.725 | 0.000 | 1.000 |
| Reactive (last observed matrix) | 27.233% | 0.091328% | 0.412 | 0.535 | 2.192 | 1.002 |
| Proactive (forecast) | 28.170% | 0.094280% | 0.416 | 0.519 | 2.212 | 1.002 |
| Hybrid (max of forecast and last observation) | 20.362% | 0.055664% | 0.400 | 0.509 | 2.763 | 1.003 |
| Oracle (true future matrix) | 10.681% | 0.016479% | 0.405 | 0.484 | 2.191 | 1.002 |

At the matched threshold, proactive routing does **not** improve the aggregate 45%-threshold exceedance rate or proxy versus reactive routing; it is slightly worse on both and has a marginally lower p95 peak utilization. Reroute counts and path stretch are nearly equal. The defensible predictive result is link-load estimation: RMSE is 1.020% of capacity for the forecast versus 1.305% for last observation, a 21.8% reduction. With the same 45% event threshold for both estimates and ground truth, F1 is 0.876 versus 0.858. These are forecast/detection results, not evidence of network QoS improvement. No packet-level delay, loss, or throughput was measured.

The matched-threshold test replay is `res_STACK_GEANT_CAP/routing_results_common_thr045_k5_overflow.json`; current summary and CSV tables are in the same folder. The test set was examined during iterative development and is not an untouched benchmark. Treat every test result here as exploratory.

### Routing experiment history and comparability

The routing setup changed during development. Keep these experiments separate and do not present their cross-setup differences as a controlled model improvement:

| Experiment | Capacity basis | Routing setup | Threshold comparison | Result file |
|---|---|---|---|---|
| Earlier stress test | Uniform capacity estimated from training traffic | Earlier route optimizer/settings | 70% | `res_STACK_BEST/routing_results.json` |
| Per-link routing v1 | Archived GÉANT per-link capacities and IGP weights | 3 paths, minmax | Reactive 70%, proactive 50% | `res_STACK_GEANT_CAP/routing_results_k3_minmax.json` |
| Validation sweep selection | Archived per-link capacities | 3/5/8 paths; minmax/overflow; 40–60% forecast triggers | Selection used validation; test used selected proactive 45% vs reactive 70% | `routing_capacity_model_validation.json`, `res_STACK_GEANT_CAP/routing_capacity_model_results.json` |
| Matched-threshold replay | Same archived per-link capacities | 5 paths, overflow | All controllers 45% | `res_STACK_GEANT_CAP/routing_results_common_thr045_k5_overflow.json` |

The validation sweep considered 30 controller configurations, but it did not make the entire iterative project process preregistered. Since test results were viewed while changing routing setups, do not claim confirmatory test-set gains or choose only the most favorable setup. The archived uniform-capacity stress result is a different scenario and is not comparable to the per-link-capacity experiment.

### Fair trigger-grid replay

`fair_routing_eval.py` replays reactive and proactive routing at every common action trigger (with the 70% reporting threshold fixed), then selects each deployable policy's trigger on validation using one rule and a 3-flow/step reroute budget. With triggers 45%, 55%, and 70%, both reactive and proactive were selected at 55% on validation. On the previously inspected test period:

| Common action trigger | Policy | Test steps >70% | Traffic proxy >70% | Mean peak utilization | Flows rerouted / step |
|---:|---|---:|---:|---:|---:|
| 45% | Reactive | 0.000% | 0.0000% | 0.412 | 2.19 |
| 45% | Proactive | 0.062% | 0.0002% | 0.416 | 2.21 |
| 55% | Reactive | 0.437% | 0.0014% | 0.453 | 0.44 |
| 55% | Proactive | 0.125% | 0.0011% | 0.451 | 0.33 |
| 70% | Reactive | 5.434% | 0.0043% | 0.485 | 0.08 |
| 70% | Proactive | 4.560% | 0.0039% | 0.483 | 0.10 |

At the validation-selected 55% trigger, proactive minus reactive was **-0.312 percentage points** for steps above 70% (95% moving-block bootstrap interval **[-0.625, -0.062]**), **-0.000283 percentage points** for the above-70% traffic proxy (interval **[-0.000775, -0.000017]**), and **-0.105 flows/step** (interval **[-0.190, -0.040]**). The matched 45% comparison slightly favors reactive on the point estimates; at 70%, the steps interval includes zero. This is more balanced evidence than the earlier unequal-trigger comparison, but remains exploratory because the test period was already examined during development and validation forecast selection may be optimistic.

The evaluator writes the full trigger grid, selected validation thresholds, and paired intervals to `res_STACK_GEANT_CAP/fair_routing_comparison.json`. `fair_routing_eval.log` preserves the formatted console tables.

### Synthetic capacity stress sensitivity

To examine a genuinely overloaded regime, the same fair evaluator was rerun with every archived capacity multiplied by 0.50 and 0.35. These are **synthetic sensitivity scenarios**, not estimates of actual GÉANT capacity. The trigger grid and 3-flow/step budget were held fixed. No trigger met that reroute budget on validation in either scenario; the evaluator's 70% selection is therefore only its fewest-reroutes fallback, not a feasible budget-selected operating point.

At matched 70% triggers, the above-capacity traffic proxy fell from 0.0134% to 0.0105% at 50% capacity and from 0.1971% to 0.1511% at 35% capacity when using the forecast instead of last observation. The step rate above 70% was nearly tied at that trigger, with bootstrap intervals including zero; proactive also required more reroutes (4.89 vs. 4.63 and 9.49 vs. 8.56 per interval). At matched 45% and 55% triggers, the forecast reduced above-70% steps and proxy with bootstrap intervals below zero, but required roughly 0.65–1.45 additional flow reroutes per step. These results suggest the forecast can help distribute load under assumed stress, at a control-cost trade-off; they do not establish performance for real GÉANT capacities or packet loss.

All tested trigger rows, paired intervals, and capacity-above-traffic proxies are summarized in `res_STACK_GEANT_CAP/capacity_stress_comparison.md`; raw evaluator JSON is under `res_STACK_GEANT_CAP/stress_50pct/` and `stress_35pct/`.

### Reroute-cost and multi-commodity-flow experiments

`routing_advanced_experiments.py` compares three methods at a common 55% action threshold and 70% reporting threshold: independent greedy overflow routing, a persistent greedy optimizer with a route-change penalty, and a capacity-aware LP that can split each OD demand across candidate paths. Penalties were selected on validation at both the archived capacity scale and a 35% synthetic capacity scale. For the LP, one path-cost penalty was selected jointly for reactive and forecast inputs. The full results are in `res_STACK_GEANT_CAP/advanced_routing/advanced_routing_results.json` and the table is `res_STACK_GEANT_CAP/advanced_routing/advanced_routing_summary.md`.

Validation selected a zero explicit route-change penalty at both scales; positive penalties worsened the selected 70%-threshold traffic metric. The benefit on test therefore comes from **warm-starting from the previous route assignment**, not from a positive penalty value. On archived capacities, this persistent policy reduced steps above 70% to zero for both reactive and proactive inputs, with fewer than 0.01 route changes per interval; no full-capacity overload existed. At 35% synthetic capacity, it reduced the above-capacity traffic proxy from 0.1964% to 0.0290% for reactive input and from 0.1504% to 0.0290% for proactive input. It kept many flows on alternate paths and increased mean IGP stretch to about 2.5%.

The capacity-aware multi-commodity-flow (MCF) LP uses the same 55% trigger but permits fractional splitting of OD flows. Validation selected a shared 0.01 IGP-cost penalty; no tested MCF setting met the 3 split-flow-equivalent budget. At 35% capacity, MCF lowered mean peak utilization (0.780 reactive, 0.760 proactive) and the above-capacity proxy (0.1522% reactive, 0.1215% proactive) versus the greedy baseline, but about 71–78 OD flows used alternate paths per interval, with 19–26 split-flow equivalents changing. It also had a higher >70% traffic proxy than greedy routing. Treat it as an idealized splittable-flow benchmark, not a directly deployable single-path result.

These advanced-policy metrics are point estimates without bootstrap intervals. The reduced-capacity case is synthetic and test results remain exploratory.

### Separate route-stability experiment

This policy carries each flow's previous path forward and changes it only when predicted peak-utilization gain clears a hysteresis margin. Margins were selected on validation separately for reactive and proactive input using the selected 5-path overflow model and 45% proactive trigger. Selection minimized route changes while limiting added high-utilization intervals to 0.125 percentage points, added above-trigger proxy to 0.002 percentage points, and mean IGP-cost stretch to 1.02. Selected margins were 0.40 for reactive input and 0.01 for proactive input.

| Input policy | Route handling | Test intervals above 70% | Route changes / interval | Flows rerouted / interval | Mean IGP-cost stretch |
|---|---|---:|---:|---:|---:|
| Reactive | Independent each interval | 5.434% | 0.10493 | 0.081 | 1.00002 |
| Reactive | **Persistent, hysteresis 0.40** | **0.375%** | **0.00187** | 2.012 | 1.00064 |
| Proactive | Independent each interval | 0.062% | 0.94129 | 2.212 | 1.00201 |
| Proactive | **Persistent, hysteresis 0.01** | **0.000%** | **0.00500** | 7.279 | 1.00878 |

Persistent routing reduced test route changes by 98.2% for reactive input and 99.5% for proactive input. It also lowered high-utilization intervals and brought the proactive above-trigger proxy to zero. This comes with more rerouted flows (7.28 vs. 2.21 per interval) and about 0.88% mean IGP path-cost stretch versus 0.20% for independent proactive routing. No policy exceeded full link capacity. This is a stability/utilization trade-off, not evidence of lower packet loss. The test set was examined during development, so results remain exploratory.

The validation sweep is `routing_stability_validation.json`; test metrics are `res_STACK_GEANT_CAP/routing_stability_results.json`. The separate chart is `presentation_capacity/routing_stability.png`. It uses validation-selected 70% reactive and 45% proactive input thresholds, so compare independent vs persistent within each input type only; it is not the matched-45% experiment above.

### Routing improvement priorities

1. **Capacity-aware replay — implemented.** Use verified link-specific capacity and IGP weights and keep all mappings auditable in `geant_link_capacities.csv`.
2. **Decision-level validation — implemented.** Select routing triggers by utilization and traffic-above-trigger outcomes, not forecasting RMSE alone. This is consistent with research showing TM error size alone does not determine routing quality.
3. **Fresh chronological holdout — still needed.** The current test set was viewed during development; do not present it as an untouched independent confirmation.
4. **Packet-level network simulation — still needed for QoS claims.** Add throughput, delay, and packet loss measurements before describing the overload proxy as network performance.

Sources: [TOTEM traffic-matrix and topology archive](https://totem.run.montefiore.uliege.be/datatools.html); [Uhlig et al., public intradomain traffic matrices](https://doi.org/10.1145/1111322.1111341); [IGen archived 25-Nov-2004 GÉANT topology with per-link bandwidth and IGP metric](https://igen.sourceforge.net/downloads/geant-20041125.gml.html); [Roughan, Thorup & Zhang, routing with estimated traffic matrices](https://doi.org/10.1145/948205.948237); [IETF RFC 9522, Internet traffic-engineering principles](https://www.rfc-editor.org/rfc/rfc9522.html).

## Files to use

- `res_STACK_BEST/metrics.json` — validation-CV-selected stack and test metrics.
- `res_STACK_BEST/pred_test.npy`, `true_test.npy`, and `last_observed_test.npy` — aligned forecast, target, and reactive input matrices.
- `res_STACK_GEANT_CAP/routing_results.json` — per-link capacity/IGP-aware routing comparison on the test period.
- `res_STACK_GEANT_CAP/routing_results_common_thr045_k5_overflow.json` — archived matched-threshold 45% replay.
- `geant_link_capacities.csv` — capacity and IGP metric for each link in matrix node order.
- `routing_capacity_model_validation.json` — validation-only routing sweep over path count, objective, and trigger.
- `res_STACK_GEANT_CAP/routing_capacity_model_results.json` — held-out replay for the validation-selected setup.
- `routing_stability_validation.json` — validation-only stability-margin sweep.
- `res_STACK_GEANT_CAP/routing_stability_results.json` — separate persistent-route test replay.
- `res_STACK_GEANT_CAP/routing_summary.md` — compact presentation-ready routing tables.
- `res_STACK_GEANT_CAP/routing_comparison.csv` and `routing_stability_comparison.csv` — spreadsheet-ready tables.
- `res_STACK_GEANT_CAP/capacity_stress_comparison.md` — sensitivity results for 50% and 35% capacity assumptions.
- `res_STACK_GEANT_CAP/advanced_routing/advanced_routing_results.json` — validation sweeps and advanced test comparisons.
- `res_STACK_GEANT_CAP/advanced_routing/advanced_routing_summary.md` — compact advanced-policy tables.
- `presentation_capacity/forecast_errors.png` — absolute forecast errors.
- `presentation_capacity/normalized_rmse.png` — per-OD normalized RMSE comparison.
- `presentation_capacity/forecast_example.png` — actual, predicted, and last-observed aggregate traffic over the first test day.
- `presentation_capacity/routing_stress_test.png` — intervals above the common 45% controller threshold and traffic proxy under archived per-link capacities.
- `presentation_capacity/routing_stability.png` — separate comparison of route stability, utilization, reroutes, and IGP path cost.
- `presentation/test_metrics.csv` — forecast metrics for charts or tables.

## Reproduce the result

Run these commands from the project folder. The CNN-LSTM checkpoint already in `res_S4L/` is used for the reported forecast; rerunning its 40-epoch CPU training can take a long time. The shared LSTM candidate is saved in `res_SHARED_LSTM_S8/`.

```powershell
py linear_forecast.py --data geant_final.npy --nodes 23 --window 12 --horizon 4 --times geant_times.npy --step-min 15 --season 96 --mask-diag --scaling mean --selection normalized_rmse --per-flow-selection --out res_PERFLOW_MEAN
py flow_tree.py --data geant_final.npy --times geant_times.npy --nodes 23 --window 12 --horizon 4 --step-min 15 --season 96 --scale range --train-stride 4 --out res_TREE_RANGE
py train.py --data geant_final.npy --nodes 23 --window 12 --horizon 4 --times geant_times.npy --step-min 15 --epochs 12 --batch 64 --patience 4 --season 96 --shared --local-scale --mask-diag --only-plain --train-stride 8 --seed 0 --out res_SHARED_LSTM_S8
py meta_stack_best.py
New-Item -ItemType Directory -Force res_STACK_GEANT_CAP | Out-Null
Copy-Item res_STACK_BEST/pred_test.npy,res_STACK_BEST/true_test.npy,res_STACK_BEST/last_observed_test.npy,res_STACK_BEST/metrics.json -Destination res_STACK_GEANT_CAP/ -Force
py routing_capacity_model_sweep.py
py routing.py --results res_STACK_GEANT_CAP --nodes 23 --topo geant --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --thr 0.45 --lag 4 --forecast-label "same 0.45 trigger for all"
py routing_stability_sweep.py
py make_presentation_figures.py --results res_STACK_GEANT_CAP --out presentation_capacity
py routing_report.py
```

For new CNN-LSTM runs, `train.py` now saves validation forecasts, test-time last observations, and input histories so blending and routing comparisons can be repeated without reconstructing them from test labels. `linear_forecast.py` performs its model selection on validation data only.

## Suggested presentation wording

> With the same 45% controller threshold, the forecast reduced link-load RMSE from 1.305% to 1.020% of capacity versus last observation. The proactive controller did not improve the 45%-threshold exceedance rate versus reactive routing (28.17% vs 27.23%). No link exceeded full capacity in the test data under the archived per-link capacity assumptions. These exploratory results do not establish packet-loss, delay, or throughput improvements.

[^geant]: S. Uhlig et al., [“Providing public intradomain traffic matrices to the research community,” ACM SIGCOMM Computer Communication Review (2006)](https://doi.org/10.1145/1111322.1111341). The local dataset files and topology ordering are described in `dataset_info.txt` and `geant_edges.txt`.

The frequency-splitting idea from the attached literature notes is described in Hu, Liu & Huo, [“An intelligent network traffic prediction method based on Butterworth filter and CNN–LSTM,” Computer Networks (2024)](https://doi.org/10.1016/j.comnet.2024.110172). The reported ensemble combines CNN-LSTM, ridge, histogram gradient boosting, and persistence; it does not claim to reproduce that paper’s Butterworth model.
