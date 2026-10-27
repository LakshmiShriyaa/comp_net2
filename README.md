# WAN traffic forecasting and proactive routing

This project predicts short-term traffic matrices for the 23-node GÉANT network, estimates link load, and compares static, reactive, and proactive routing. The detailed experiment, assumptions, measured results, and presentation wording are in [PROJECT_RESULTS.md](PROJECT_RESULTS.md).

## Current result

The validation-CV-selected per-OD stack forecasts one hour ahead. On the examined test period it improves over last-value persistence by 19.1% per-OD normalized RMSE, 8.3% raw RMSE, and 8.6% MAE. Chronological validation-fold improvement is 18.0%, though some base-model choices used the full validation split. This is close to, but below, the 20% target. The test period was inspected during model development, so confirm the result on a fresh chronological holdout before treating it as an independent benchmark.

For routing, an archived 2004 GÉANT topology supplies assumed per-link capacities and IGP weights, uniquely matched to the anonymized graph. With 5 candidate paths, the overflow objective, and a common 45% trigger, proactive routing did not improve the 45%-threshold exceedance rate over reactive routing (28.17% vs. 27.23%). The forecast did lower link-load RMSE from 1.305% to 1.020% of capacity. No link exceeded full capacity in the observed test data under these assumptions. This test set was examined during iterative development; results are exploratory and do not establish a QoS gain. See [PROJECT_RESULTS.md](PROJECT_RESULTS.md) for provenance and caveats.

In the non-persistent, capped comparison, the forecast-driven router reduced mean peak utilization from 0.398 to 0.373 at the validation-selected 55% trigger. A follow-up matched-persistence evaluation selected hysteresis 0.30 for reactive and 0.10 for proactive at that trigger. Both persistent schemes had zero intervals above 70%; the paired mean-peak difference was -0.0187 with 95% interval [-0.0426, +0.0039], which includes zero. At 70%, the persistent forecast scheme had higher mean peak utilization. The clearest result is that persistence reduces route churn for both schemes; a forecast benefit beyond persistence is not established. See `res_STACK_GEANT_CAP/persistence_eval/fair_routing_comparison.json` and the caveats in [PROJECT_RESULTS.md](PROJECT_RESULTS.md).

A separate route-stability experiment used validation-selected thresholds (70% reactive, 45% proactive); within its proactive comparison, persistence reduced route changes by 99.5% and intervals above 70% utilization to zero. It reroutes more flows (7.28 per interval) and adds 0.88% mean IGP path cost. This is a separate policy result, not a matched-threshold comparison; see the report.

## Install

Python 3, PyTorch, NumPy, NetworkX, SciPy, and Matplotlib are used by the scripts:

```powershell
py -m pip install torch numpy networkx scipy matplotlib
```

## Reproduce forecasting and routing

These commands re-fit the per-OD ridge and flow-tree models, train the shared LSTM candidate, select stack weights on validation folds, evaluate the forecast, and replay routing on the GEANT topology. They reuse the saved base-model prediction folders already in this project, including `res_S4L/`, `res_S4L_s1/`, `res_S4L_s2/`, `res_NORM_CNN/`, `res_LINEAR_SEASONAL/`, `res_LOG_FLOW/`, and `res_TREE_FULL/`; they do not retrain every neural base model from scratch.

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
py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.45 0.55 0.70 --report-thr 0.70 --budget 3 --hybrid --boot 2000 --block 96
py make_presentation_figures.py --results res_STACK_GEANT_CAP --out presentation_capacity
py routing_report.py
py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.45 0.55 0.70 --report-thr 0.70 --budget 3 --hybrid --boot 2000 --block 96 --cap-scale 0.50 --out res_STACK_GEANT_CAP/stress_50pct
py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.45 0.55 0.70 --report-thr 0.70 --budget 3 --hybrid --boot 2000 --block 96 --cap-scale 0.35 --out res_STACK_GEANT_CAP/stress_35pct
py routing_advanced_experiments.py --scales 1.0 0.35 --trigger 0.55 --report-threshold 0.70 --budget 3 --reroute-penalties 0 0.003 0.01 0.03 0.1 --mcf-penalties 0.01 0.1 --out res_STACK_GEANT_CAP/advanced_routing
py make_advanced_routing_summary.py
py routing_policy_fair_eval.py --scales 1.0 --boot 1000 --out res_STACK_GEANT_CAP/policy_hardcap_results_real.json
```

The final compact tables are `res_STACK_GEANT_CAP/routing_summary.md`, `routing_comparison.csv`, and `routing_stability_comparison.csv`. The 2004 capacity/metric mapping is in `geant_link_capacities.csv`; the previous uniform-capacity stress replay remains available in `res_STACK_BEST`. `over_trigger_traffic_proxy_%` is a simulator proxy, not measured packet loss.

## Fair trigger comparison

To compare reactive and proactive routing at common thresholds and choose each controller's threshold on validation using the same three-flow-per-step reroute budget, run:

```powershell
py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.45 0.55 0.70 --report-thr 0.70 --budget 3 --hybrid --boot 2000 --block 96
```

This writes `res_STACK_GEANT_CAP/fair_routing_comparison.json` and prints the same-trigger grid plus paired 95% moving-block bootstrap intervals. Trigger selection uses validation forecasts, which can be optimistic if those forecasts were tuned on the same validation targets. The test set has also been inspected during development, so treat test comparisons as exploratory.

To compare persistent reactive and persistent proactive routes at each matched trigger, with hysteresis chosen per scheme and trigger on validation, run:

```powershell
py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.45 0.55 0.70 --persistence --hysteresis 0.1 0.3 0.5 0.7 --out res_STACK_GEANT_CAP/persistence_eval
```

This writes `persistence_eval/fair_routing_comparison.json`; sections C and D in the console output compare persistence against independent routing and persistent proactive against persistent reactive. The test split has been inspected and the stack's validation forecasts are in-sample, so treat these as exploratory results.

For optional synthetic capacity stress sensitivity runs (not actual GÉANT capacity claims), use fixed scale factors and separate output folders:

```powershell
py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.45 0.55 0.70 --report-thr 0.70 --budget 3 --hybrid --boot 2000 --block 96 --cap-scale 0.50 --out res_STACK_GEANT_CAP/stress_50pct
py fair_routing_eval.py --results res_STACK_GEANT_CAP --val-results res_STACK_BEST --nodes 23 --edges geant_edges.txt --link-metadata geant_link_capacities.csv --k 5 --optimizer overflow --triggers 0.45 0.55 0.70 --report-thr 0.70 --budget 3 --hybrid --boot 2000 --block 96 --cap-scale 0.35 --out res_STACK_GEANT_CAP/stress_35pct
py make_capacity_stress_summary.py
```

See `res_STACK_GEANT_CAP/capacity_stress_comparison.md`. If no trigger meets the reroute budget under stress, the evaluator reports the fewest-reroutes fallback; it should not be described as satisfying that budget.

## Advanced routing comparison

To tune persistent route-change penalties and compare a capacity-aware fractional MCF baseline at the same 55% action threshold, run:

```powershell
py routing_advanced_experiments.py --scales 1.0 0.35 --trigger 0.55 --report-threshold 0.70 --budget 3 --reroute-penalties 0 0.003 0.01 0.03 0.1 --mcf-penalties 0.01 0.1 --out res_STACK_GEANT_CAP/advanced_routing
py make_advanced_routing_summary.py
```

The MCF LP allows fractional OD-flow splitting and is an idealized benchmark. The comparison writes JSON and Markdown under `res_STACK_GEANT_CAP/advanced_routing/`; reduced-capacity results are synthetic, and the test period is exploratory.

## Train the CNN-LSTM again

```powershell
py train.py --data geant_final.npy --nodes 23 --window 12 --horizon 4 --times geant_times.npy --step-min 15 --epochs 40 --batch 64 --patience 8 --season 96 --shared --local-scale --mask-diag --skip-plain --seed 0 --out res_CNN
```

The script saves validation forecasts, exact last-observed matrices, and input histories. Full CPU training can be slow; use a PyTorch CUDA environment when available. Keep the chronological split and only use validation data to select model or controller settings.
