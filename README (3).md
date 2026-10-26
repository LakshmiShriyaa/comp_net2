# CNN-LSTM traffic forecasting + proactive routing for a WAN

Centralized SDN-style design: the controller collects traffic matrices, a CNN-LSTM predicts the
next matrix, predicted congestion is detected, and flows are rerouted before it happens.

```
traffic matrices -> CNN-LSTM -> predicted matrix -> link loads -> congestion check -> reroute
```

## Files

| File | Purpose |
|---|---|
| `data.py` | Load Abilene/GEANT matrices (or generate synthetic ones), scale, window, chronological split |
| `model.py` | `CNNLSTM` (the proposed model) and `PlainLSTM` (baseline), PyTorch |
| `train.py` | Trains both, compares against naive and moving-average baselines (MAE, RMSE, MAPE), saves predictions |
| `routing.py` | Congestion detection and greedy proactive rerouting; compares static, reactive, proactive, oracle |
| `export_ns3.py` | Turns controller decisions into CSV inputs for NS-3 |
| `ns3_wan_sim.cc` | NS-3 Abilene simulation driven by those CSVs |

## Setup

```
pip install torch numpy networkx
```

## 1. Train and evaluate the forecaster

Quick test on synthetic data (no download needed):

```
python train.py
```

Real data (a `.npy` shaped `(T, N, N)` or `(T, N*N)`, or a `.csv` with N*N values per row):

```
python train.py --data abilene.npy --nodes 12 --window 12 --epochs 60
python train.py --data geant.npy   --nodes 23 --window 12 --epochs 60
```

Outputs go to `results/`: `pred_test.npy`, `true_test.npy`, `cnn_lstm.pt`, `metrics.json`.

## 2. Congestion detection and proactive routing

```
python routing.py --results results --nodes 12 --topo abilene
```

To force visible congestion, size capacity tighter: `--cap-pct 70 --cap-util 1.0 --thr 0.85`.
Schemes compared: static shortest path, reactive (uses last observed matrix), proactive (uses the
CNN-LSTM prediction), and oracle (uses the true matrix, an upper bound).

## 3. NS-3 simulation

```
python export_ns3.py --results results --steps 40 --link-mbps 100 --out ns3_input
```

Copy `ns3_wan_sim.cc` into your ns-3 `scratch/` folder and run it once per mode
(`static`, `reactive`, `proactive`); the exact commands are in the comment at the top of the file.
Compare packet loss, delay and throughput across the three summaries.

## Things to check before trusting results

- **Node order.** `routing.py` assumes the dataset columns follow `ABILENE_NODES`. Verify this against
  the dataset documentation. Abilene is hardcoded; GEANT needs its real link list via `--edges`
  (otherwise a made-up ring-plus-chords topology is used, which is fine for testing only).
- **Capacity.** The datasets give demands, not link capacities, so capacity is set uniformly by a
  percentile rule (`--cap-pct`, `--cap-util`). State this assumption in your report.
- **Units and sampling.** Abilene is 5-minute data and GEANT is 15-minute, so one step is not the same
  amount of time in the two datasets.
- **Metrics.** `overload_traffic_%` in `routing.py` is a proxy for dropped traffic, not simulated loss.
  Use NS-3 for real packet loss and delay.
