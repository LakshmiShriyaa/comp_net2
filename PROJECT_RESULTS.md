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

### Earlier average-budget persistent routing experiment

The independent overflow router rebuilds routes from shortest paths at every interval. The new persistent variant starts from its previous route assignment and moves flows only when doing so lowers squared utilization excess above the action trigger. `routing_policy_fair_eval.py` selects one trigger and penalty jointly for reactive and forecast inputs on validation, subject to an average budget of 3 actual whole-flow route changes per step. Both inputs then use the same capacity mapping, five paths, trigger, penalty, and routing policy on test. This is an average budget; it is not a hard per-interval limit.

At archived capacities, validation selected the persistent overflow policy with a 55% trigger and zero additional penalty. On the inspected test period, persistent reactive routing reduced steps above 70% from 0.437% to 0.000%, the above-70% traffic proxy from 0.001363% to zero, mean peak utilization from 0.453 to 0.402 (11.3% lower), and actual route changes from 0.335 to 0.0069 per step (98% fewer). The paired 95% moving-block intervals for these differences excluded zero. No physical overload occurred under either policy. The trade-off was more flows remaining on alternate paths (6.64 vs. 0.44 per step) and a 0.10% increase in mean IGP-cost stretch (1.0002 to 1.0012). These are utilization and control-cost results, not packet-loss or QoS measurements.

| Capacity scale | Reactive independent steps >70% | Persistent reactive steps >70% | Reactive independent >capacity proxy | Persistent reactive >capacity proxy | Mean peak: independent → persistent | Route changes/step: independent → persistent | IGP stretch: independent → persistent |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100% archived | 0.437% | 0.000% | 0.0000% | 0.0000% | 0.453 → 0.402 | 0.335 → 0.007 | 1.0002 → 1.0012 |
| 50% synthetic | 34.354% | 29.044% | 0.0102% | 0.0007% | 0.653 → 0.611 | 2.488 → 0.058 | 1.0090 → 1.0232 |
| 35% synthetic | 74.391% | 65.209% | 0.2618% | 0.0289% | 0.925 → 0.863 | 1.026 → 0.138 | 1.0091 → 1.0259 |

The 50% and 35% rows are synthetic capacity stress tests, not observed GÉANT capacity conditions. Their paired bootstrap intervals and full metrics are in `res_STACK_GEANT_CAP/policy_fair_results_stress.json` and `.md`.

With the persistent optimizer held fixed at the same 55% trigger, replacing reactive input with the full forecast did not establish an additional benefit on the archived-capacity test: both policies had zero steps above 70%; the mean-peak difference was -0.0187 with 95% interval [-0.0432, +0.0033], while IGP stretch increased by 0.00293. Validation-tuned blends selected weights 0.50 at archived capacity, 0.25 at 50%, and 0.10 at 35%; they did not produce a consistent test improvement over reactive input. Full results are in `res_STACK_GEANT_CAP/forecast_blend_results.json` and `.md`.

This average-budget experiment suggested a routing improvement, but did not constrain the maximum route changes in an individual interval. The stricter hard-cap comparison below supersedes it for the main claim. The test period was examined during development, so all test comparisons remain exploratory; a fresh chronological holdout is still required for confirmatory claims.

### Matched hard-cap routing comparison

To enforce the same operational limit at every interval, all candidate controllers were capped at **three final whole-flow route changes per step**. For the independent overflow baseline, the optimizer proposes a route assignment and the evaluator commits at most three beneficial changes relative to the previous assignment. Persistent policies use the same three-change cap directly. Trigger and policy choices are made using validation only; at archived capacity, the shared selected trigger was 55%. Test comparisons use paired 96-step moving-block bootstrap intervals (1,000 replicates) and remain exploratory because the test period has been inspected during development.

At archived capacities, the validation-selected controller was independent overflow for both input types. Supplying the one-hour traffic forecast instead of the lagged reactive observation reduced mean peak link utilization from **0.398 to 0.373** (absolute change -0.0248; paired 95% interval [-0.0469, -0.0124]), about a **6.2% relative reduction**. Test intervals with any link above 70% went from **0.062% to 0.000%** (one interval to none), but that event-count interval includes zero. The above-70% traffic proxy also fell by a tiny amount; true capacity overload remained zero for both. Average route changes were 0.0081 vs. 0.0044 per step, and the maximum was at most three for every step. Mean IGP path-cost stretch increased slightly from 1.0034 to 1.0045 (paired difference +0.00118; 95% interval [+0.00013, +0.00281]). Thus the concrete win is lower peak utilization under the matched hard cap, not demonstrated congestion or packet-loss removal.

| Capacity scenario | Validation-selected policy and trigger | Reactive baseline steps >70% | Selected reactive policy steps >70% | Reactive baseline >capacity proxy | Selected reactive policy >capacity proxy | Mean peak utilization, baseline → selected | IGP stretch, baseline → selected |
|---|---|---:|---:|---:|---:|---:|---:|
| Archived capacity | Independent overflow, 55% | 0.062% | 0.062% | 0.0000% | 0.0000% | 0.398 → 0.398 | 1.0034 → 1.0034 |
| 50% synthetic | Persistent overflow, 55% | 29.794% | 29.044% | 0.00076% | 0.00073% | 0.613 → 0.611 | 1.0163 → 1.0232 |
| 35% synthetic | Persistent overflow, 45% | 65.459% | 65.209% | 0.02868% | 0.02915% | 0.864 → 0.863 | 1.0270 → 1.0253 |

On the two synthetic stress scales, persistence gave mixed and small changes; at 35% the above-capacity proxy increased slightly. The forecast itself was not consistently helpful under stress. These artificial capacity reductions are sensitivity tests and are not measured GÉANT operating conditions. Full tables and bootstrap intervals are in `res_STACK_GEANT_CAP/policy_hardcap_results_real.md`, `policy_hardcap_results_50pct.md`, and `policy_hardcap_results_35pct.md`.

### Matched persistence: forecast versus reactive

The follow-up evaluator `fair_routing_eval.py --persistence` replays both inputs with and without persistence at the same three triggers, selects each hysteresis on validation using one shared rule, and computes paired 96-step moving-block intervals on test. Validation selected a 55% trigger for both schemes; it selected hysteresis 0.30 for reactive and 0.10 for proactive at that trigger. The stack's validation predictions are in-sample, so this selection may favor proactive routing. The test set was also inspected during development.

At the selected 55% trigger, persistent reactive and persistent proactive both had **0.000%** of intervals above 70% and zero above-70% traffic proxy. The proactive-minus-reactive mean-peak difference was -0.0187 utilization (95% interval [-0.0426, +0.0039]), which includes zero. Actual route changes differed by -0.0031 per interval ([-0.0081, +0.0006]), also inconclusive. Proactive used 3.36 fewer flows off their default path on average, but its mean IGP stretch was higher by 0.00293 (95% interval [+0.00076, +0.00509]). Thus the earlier 6.2% mean-peak reduction describes forecast versus reactive input under the capped non-hysteretic comparison; it does **not** establish a forecast benefit after both inputs receive validation-selected persistence.

| Same trigger | Reactive h | Proactive h | Δ steps >70% (pp) [95% CI] | Δ mean peak [95% CI] | Δ route changes/step [95% CI] | Δ flows off default/step [95% CI] | Δ IGP stretch [95% CI] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 45% | 0.30 | 0.10 | 0.000 [0.000, 0.000] | +0.0007 [-0.0001, +0.0019] | -0.0031 [-0.0087, +0.0006] | -0.863 [-0.998, -0.687] | -0.00024 [-0.00029, -0.00019] |
| **55% (validation-selected)** | **0.30** | **0.10** | **0.000 [0.000, 0.000]** | **-0.0187 [-0.0426, +0.0039]** | **-0.0031 [-0.0081, +0.0006]** | **-3.361 [-4.515, -2.062]** | **+0.00293 [+0.00076, +0.00509]** |
| 70% | 0.50 | 0.10 | 0.000 [0.000, 0.000] | +0.0070 [+0.0042, +0.0107] | +0.0012 [0.0000, +0.0037] | +1.913 [+1.848, +2.002] | +0.00004 [+0.00003, +0.00005] |

At 70%, persistent proactive had significantly higher mean peak utilization than persistent reactive. Across matched triggers, the clearest consistent effect is persistence versus each scheme's independent routing: it reduced actual route changes, and at 55% it reduced above-70% intervals for both reactive and proactive inputs. This is exploratory evidence from the inspected period, not a confirmatory forecast advantage. Full validation selections, all six paired metrics, and intervals are in `res_STACK_GEANT_CAP/persistence_eval/fair_routing_comparison.json`.

As a sensitivity check for the unequal validation-selected hysteresis values, Section E reran proactive-minus-reactive using the **same fixed hysteresis** for both schemes (0.10 and 0.30; these settings were not selected on test). At 55%, the mean-peak difference was -0.0187 with interval [-0.0426, +0.0039] for either fixed setting, so it still includes zero; at 70%, it was +0.0070 with interval [+0.0042, +0.0107] for either setting. Both schemes again had zero intervals above 70% at these triggers, while proactive path stretch remained higher at 55%. At 45%, the estimated mean-peak difference was a tiny +0.0008 for both settings; its interval included zero at h=0.10 and just excluded zero at h=0.30. Given the small size and multiple comparisons, do not treat that borderline result as evidence of a forecast disadvantage. Overall, the main 55% and 70% conclusions are not explained by the different selected hysteresis values. Full Section E results are in `res_STACK_GEANT_CAP/persistence_eval_fixed/fair_routing_comparison.json`.

For clarity, this evaluator's `--budget` is the average number of flows assigned to non-default paths, used only when selecting a trigger. It is not a cap on how many route assignments can change in one interval. Actual route changes are measured separately in the output.

The run examines multiple triggers, persistence settings, and metrics without a multiplicity correction. Treat individual intervals as descriptive; the inspected test period and in-sample validation forecasts further limit confirmatory interpretation.

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
- `res_STACK_GEANT_CAP/policy_fair_results_real.json` and `.md` — validation-selected persistent-router comparison at archived capacities with paired bootstrap intervals.
- `res_STACK_GEANT_CAP/policy_fair_results_stress.json` and `.md` — the same comparison at synthetic 50% and 35% capacity scales.
- `res_STACK_GEANT_CAP/forecast_blend_results.json` and `.md` — validation-tuned forecast-weight sweep under persistent routing.
- `res_STACK_GEANT_CAP/policy_hardcap_results_real.json` and `.md` — main validation-selected comparison with a hard cap of three actual route changes per interval.
- `res_STACK_GEANT_CAP/policy_hardcap_results_50pct.json` and `.md`, and `policy_hardcap_results_35pct.json` and `.md` — synthetic stress sensitivity under the same hard cap.
- `res_STACK_GEANT_CAP/persistence_eval/fair_routing_comparison.json` — matched-trigger persistence selection and paired reactive/proactive comparison.
- `res_STACK_GEANT_CAP/persistence_eval_fixed/fair_routing_comparison.json` — fixed-hysteresis sensitivity (Section E), using identical hysteresis for reactive and proactive.
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
py routing_policy_fair_eval.py --scales 1.0 --out res_STACK_GEANT_CAP/policy_fair_results_real.json
py routing_policy_fair_eval.py --scales 0.50 0.35 --out res_STACK_GEANT_CAP/policy_fair_results_stress.json
py routing_forecast_blend_sweep.py
py routing_policy_fair_eval.py --scales 1.0 --boot 1000 --out res_STACK_GEANT_CAP/policy_hardcap_results_real.json
py routing_policy_fair_eval.py --scales 0.50 --triggers 0.55 --boot 1000 --out res_STACK_GEANT_CAP/policy_hardcap_results_50pct.json
py routing_policy_fair_eval.py --scales 0.35 --triggers 0.45 --boot 1000 --out res_STACK_GEANT_CAP/policy_hardcap_results_35pct.json
py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.45 0.55 0.70 --persistence --hysteresis 0.1 0.3 0.5 0.7 --fixed-hysteresis 0.10 0.30 --out res_STACK_GEANT_CAP/persistence_eval_fixed
```

`routing.py` also exposes persistent overflow and peak-minimizing objectives for direct runs; `--max-route-changes 3` enforces a hard cap. `routing_policy_fair_eval.py` applies the same hard cap to the independent baseline and persistent candidates, selects policies on validation data, then writes paired test intervals.

For new CNN-LSTM runs, `train.py` now saves validation forecasts, test-time last observations, and input histories so blending and routing comparisons can be repeated without reconstructing them from test labels. `linear_forecast.py` performs its model selection on validation data only.

## Suggested presentation wording

> With the same 45% controller threshold, the forecast reduced link-load RMSE from 1.305% to 1.020% of capacity versus last observation. The proactive controller did not improve the 45%-threshold exceedance rate versus reactive routing (28.17% vs 27.23%). No link exceeded full capacity in the test data under the archived per-link capacity assumptions. These exploratory results do not establish packet-loss, delay, or throughput improvements.

[^geant]: S. Uhlig et al., [“Providing public intradomain traffic matrices to the research community,” ACM SIGCOMM Computer Communication Review (2006)](https://doi.org/10.1145/1111322.1111341). The local dataset files and topology ordering are described in `dataset_info.txt` and `geant_edges.txt`.

The frequency-splitting idea from the attached literature notes is described in Hu, Liu & Huo, [“An intelligent network traffic prediction method based on Butterworth filter and CNN–LSTM,” Computer Networks (2024)](https://doi.org/10.1016/j.comnet.2024.110172). The reported ensemble combines CNN-LSTM, ridge, histogram gradient boosting, and persistence; it does not claim to reproduce that paper’s Butterworth model.
