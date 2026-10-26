"""Data pipeline: load / generate traffic matrices, clean, normalize, window, split.

Traffic matrix array shape everywhere: (T, N, N)
    T = time steps, N = nodes, entry [t, i, j] = traffic from node i to node j.
"""
import os
import numpy as np



def tod_features(d, window: int, times=None, step_min: int = 15):
    """Time-of-day input for every step of every window: (sin, cos) of the position in the day."""
    bounds = {"train": 0, "val": d["t_end"], "test": d["v_end"]}
    out = {}
    for name, lo in bounds.items():
        idx = lo + np.asarray(d[name + "_s"])[:, None] + np.arange(window)[None, :]
        minutes = np.asarray(times)[idx] if times is not None else idx * step_min
        phase = 2.0 * np.pi * (minutes % 1440) / 1440.0
        out[name] = np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)
    return out
# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_tm(path: str, n_nodes: int) -> np.ndarray:
    """Load a traffic-matrix file and return an array of shape (T, N, N).

    Supported formats:
      .npy  : array already shaped (T, N, N) or (T, N*N)
      .csv / .txt : one row per time step, N*N values per row
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".csv", ".txt"):
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if arr.ndim == 2:
        if arr.shape[1] != n_nodes * n_nodes:
            raise ValueError(
                f"Expected {n_nodes * n_nodes} columns for {n_nodes} nodes, got {arr.shape[1]}"
            )
        arr = arr.reshape(-1, n_nodes, n_nodes)
    if arr.ndim != 3 or arr.shape[1:] != (n_nodes, n_nodes):
        raise ValueError(f"Unexpected shape {arr.shape}")
    return arr.astype(np.float32)


def synthetic_tm(T: int = 4000, n_nodes: int = 12, steps_per_day: int = 288, seed: int = 0) -> np.ndarray:
    """Synthetic WAN-like traffic matrices (gravity model + daily/weekly cycles + noise).

    Pipeline test only. Use load_tm(...) on Abilene / GEANT for real results.
    """
    rng = np.random.default_rng(seed)
    n = n_nodes
    out_w = rng.uniform(0.5, 2.0, n)
    in_w = rng.uniform(0.5, 2.0, n)
    base = np.outer(out_w, in_w) * 100.0
    np.fill_diagonal(base, 0.0)

    phase = rng.uniform(0, 2 * np.pi, (n, n))
    t = np.arange(T)[:, None, None]
    daily = 1.0 + 0.5 * np.sin(2 * np.pi * t / steps_per_day + phase)
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * t / (7 * steps_per_day))
    noise = rng.lognormal(mean=0.0, sigma=0.12, size=(T, n, n))

    bursts = np.ones((T, n, n))
    for _ in range(T // 100):
        s = rng.integers(0, T - 20)
        i, j = rng.integers(0, n, 2)
        bursts[s:s + rng.integers(5, 20), i, j] *= rng.uniform(1.5, 3.0)

    tm = base * daily * weekly * noise * bursts
    tm[:, np.arange(n), np.arange(n)] = 0.0
    return tm.astype(np.float32)


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def despike(tm: np.ndarray, factor: float = 100.0, win: int = 97):
    """Safety net for GROSSLY corrupt time steps (e.g. 1e9-1e11 kbps values).

    A step is flagged when its largest demand is more than `factor` times the
    median of the per-step maxima in a +/- win//2 neighbourhood. Flagged steps
    are replaced by linear interpolation between the nearest unflagged steps.
    Only unusually HIGH steps are flagged; dips are left alone.

    factor=100 is deliberately conservative: a lower value (20) wrongly flags
    ~88 plausible 5-7 Gbps steps in Abilene. It will NOT catch milder corruption
    such as the 1e7-4e7 block at GEANT steps 5137-5148; inspect those yourself.

    Returns (cleaned copy, list of repaired step indices) so the repair can be
    reported in your write-up.
    """
    T = len(tm)
    step_max = tm.reshape(T, -1).max(axis=1)
    half = win // 2
    bad = np.zeros(T, dtype=bool)
    for t in range(T):
        med = np.median(step_max[max(0, t - half):min(T, t + half + 1)])
        bad[t] = med > 0 and step_max[t] > factor * med

    out = tm.copy()
    idx = np.where(bad)[0]
    if len(idx) == 0:
        return out, []
    good = np.where(~bad)[0]
    for t in idx:
        prev, nxt = good[good < t], good[good > t]
        if len(prev) and len(nxt):
            a, b = prev[-1], nxt[0]
            w = (t - a) / (b - a)
            out[t] = (1 - w) * tm[a] + w * tm[b]
        elif len(prev):
            out[t] = tm[prev[-1]]
        else:
            out[t] = tm[nxt[0]]
    return out, idx.tolist()


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
class MinMaxScaler:
    """Global min-max scaler fitted on the training split only (no leakage).

    With log=True the data goes through log1p first, which compresses the heavy
    tail (a few huge OD pairs vs many tiny ones). inverse() undoes both steps.
    """

    def __init__(self, log: bool = False):
        self.log = log

    def fit(self, x: np.ndarray):
        x = np.log1p(x) if self.log else x
        self.lo = float(x.min())
        self.hi = float(x.max())
        return self

    def transform(self, x):
        x = np.log1p(x) if self.log else x
        return (x - self.lo) / (self.hi - self.lo + 1e-8)

    def inverse(self, x):
        x = x * (self.hi - self.lo + 1e-8) + self.lo
        return np.maximum(np.expm1(x), 0.0) if self.log else x


def make_windows(tm: np.ndarray, window: int, horizon: int = 1):
    """X: (S, window, N, N) past matrices;  Y: (S, horizon, N, N) next matrices."""
    xs, ys = [], []
    for s in range(len(tm) - window - horizon + 1):
        xs.append(tm[s:s + window])
        ys.append(tm[s + window:s + window + horizon])
    return np.stack(xs), np.stack(ys)


def prepare(tm: np.ndarray, window: int = 12, horizon: int = 1, train_frac: float = 0.7,
            val_frac: float = 0.15, log: bool = False, mask_diag: bool = False,
            times=None, step_min: int = 15, season: int = 0):
    """Optional diagonal zeroing -> chronological split -> scale (fit on train) -> windows.

    mask_diag zeroes the node-to-itself entries. They never cross a WAN link, so
    routing ignores them, but they can dominate the forecasting error.

    times: optional int array (T,) with each step's timestamp in MINUTES. Real datasets
    have missing files (GEANT has about 620 missing steps), so consecutive rows are not
    always step_min apart. When `times` is given:
      - a window is dropped if its history + forecast steps are not exactly step_min apart
      - a TEST window is also dropped if `season` > 0 and its seasonal lookback
        (the step `season` earlier) is not exactly season * step_min earlier
    so every model is scored on the same, properly timed windows.

    Returned dict: scaler, tm (processed series, original units), t_end, v_end,
    train/val/test = (X, Y), train_s/val_s/test_s = window start index inside each
    split, stats = {split: (windows kept, windows total)}.
    """
    if mask_diag:
        tm = tm.copy()
        k = np.arange(tm.shape[1])
        tm[:, k, k] = 0.0

    T = len(tm)
    t_end = int(T * train_frac)
    v_end = int(T * (train_frac + val_frac))
    if times is not None:
        times = np.asarray(times, dtype=np.int64)
        if len(times) != T:
            raise ValueError(f"times has {len(times)} entries but the data has {T} steps")
    train, val, test = tm[:t_end], tm[t_end:v_end], tm[v_end:]

    scaler = MinMaxScaler(log=log).fit(train)
    # "tm" is the processed series in ORIGINAL units; t_end / v_end are the split points.
    out = {"scaler": scaler, "raw_test": test, "tm": tm, "t_end": t_end, "v_end": v_end, "stats": {}}
    bounds = {"train": (0, t_end), "val": (t_end, v_end), "test": (v_end, T)}
    for name, part in (("train", train), ("val", val), ("test", test)):
        X, Y = make_windows(scaler.transform(part), window, horizon)
        total = len(X)
        s_idx = np.arange(total)
        if times is not None:
            lo, hi = bounds[name]
            tp = times[lo:hi]
            span = tp[window + horizon - 1:] - tp[:total]
            ok = span == (window + horizon - 1) * step_min
            if name == "test" and season > 0:
                tgt = lo + s_idx[:, None] + window + np.arange(horizon)[None, :]   # absolute target index
                src = tgt - season
                has_src = (src >= 0).all(axis=1)
                ok &= has_src & ((times[tgt] - times[np.maximum(src, 0)]) == season * step_min).all(axis=1)
            s_idx = np.where(ok)[0]
            X, Y = X[s_idx], Y[s_idx]
        out[name] = (X.astype(np.float32), Y.astype(np.float32))
        out[name + "_s"] = s_idx
        out["stats"][name] = (len(s_idx), total)
    return out


# ----------------------------------------------------------------------------
# Evaluation helpers (torch-free, so they can be tested without PyTorch)
# ----------------------------------------------------------------------------
def metrics(pred, true, mask=None, eps=1e-6):
    """MAE, RMSE, MAPE in original traffic units for arrays shaped (S, ..., N, N).

    `mask` is an (N, N) bool array selecting the entries to score (e.g. off-diagonal).
    MAPE only scores entries above 1% of the largest true value."""
    if mask is not None:
        pred, true = pred[..., mask], true[..., mask]
    err = pred - true
    keep = true > (true.max() * 0.01)
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE_%": float((np.abs(err[keep]) / (true[keep] + eps)).mean() * 100),
    }


def rmse_by_step(pred, true, mask=None):
    """RMSE separately for each forecast step ahead (+1, +2, ...). Arrays are (S, H, N, N)."""
    return [metrics(pred[:, j:j + 1], true[:, j:j + 1], mask)["RMSE"] for j in range(true.shape[1])]


def baseline_forecasts(d, window: int, horizon: int, season: int = 0):
    """Non-learned forecasts of the test targets, shape (S, H, N, N), original units.

    Only information available when the forecast is made is used.
      Naive            : repeat the last observed matrix
      Moving average   : mean of the last 3 matrices
      Seasonal naive   : the matrix from `season` steps before the target
                         (96 = same time yesterday for 15-minute data; needs horizon <= season)
    `d` is the dict returned by prepare().
    """
    X_o = d["scaler"].inverse(d["test"][0])                     # (S, W, N, N)
    S = len(X_o)
    shape = (S, horizon) + X_o.shape[-2:]
    out = {
        "Naive (last value)": np.broadcast_to(X_o[:, -1:], shape),
        "Moving average (3)": np.broadcast_to(X_o[:, -3:].mean(axis=1, keepdims=True), shape),
    }
    if season > 0 and horizon <= season:
        idx = d["v_end"] + d["test_s"][:, None] + window + np.arange(horizon)[None, :] - season
        if idx.min() >= 0:
            out[f"Seasonal naive ({season} ago)"] = d["tm"][idx]
    return out


def norm_metrics(pred, true, lo, hi, mask=None):
    """MAE / RMSE after scaling EVERY OD pair by its own training range (per-pair min-max),
    the way the Colab notebook scaled its data. pred/true: (..., N, N) in original units;
    lo/hi: (N, N) per-pair training minimum / maximum. Pairs whose training range is zero
    (always-zero pairs such as the diagonal) contribute no error but are still counted in the mean,
    as in the notebook. Normally called with mask=None (all N*N entries)."""
    rng = hi - lo
    w = np.where(rng > 0, 1.0 / np.where(rng > 0, rng, 1.0), 0.0)
    e = (pred - true) * w
    if mask is not None:
        e = e[..., mask]
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean()))}
