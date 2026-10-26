"""Congestion detection + proactive routing, evaluated on the CNN-LSTM predictions.

Pipeline per test step t:
    predicted matrix  -> link loads (static shortest paths)
                      -> links above threshold = predicted congestion
                      -> greedily move flows to alternative k-shortest paths
    chosen routes are then applied to the TRUE traffic at that step, so the
    comparison reflects what the network would really experience.

Schemes compared
    static     : always shortest path (no prediction)
    reactive   : reroutes using the LAST observed matrix (what a non-predictive controller sees)
    proactive  : reroutes using the CNN-LSTM predicted matrix
    oracle     : reroutes using the true matrix (upper bound)

Usage
    python routing.py --results results --nodes 12 --topo abilene
"""
import argparse
import csv
import itertools
import json
import os
import numpy as np
import networkx as nx
from data import load_tm, despike

# Abilene backbone, 15 links. Node order below must match the column order of
# your dataset - CHECK THIS against the dataset documentation before real runs.
ABILENE_NODES = ["ATLAM5", "ATLAng", "CHINng", "DNVRng", "HSTNng", "IPLSng",
                 "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng"]
ABILENE_LINKS = [("ATLAM5", "ATLAng"), ("ATLAng", "HSTNng"), ("ATLAng", "IPLSng"),
                 ("ATLAng", "WASHng"), ("CHINng", "IPLSng"), ("CHINng", "NYCMng"),
                 ("DNVRng", "KSCYng"), ("DNVRng", "SNVAng"), ("DNVRng", "STTLng"),
                 ("HSTNng", "KSCYng"), ("HSTNng", "LOSAng"), ("IPLSng", "KSCYng"),
                 ("LOSAng", "SNVAng"), ("NYCMng", "WASHng"), ("SNVAng", "STTLng")]


# ----------------------------------------------------------------------------
# Topology and path tables
# ----------------------------------------------------------------------------
def build_topology(name: str, n: int, edge_file: str = None) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if edge_file:
        with open(edge_file) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    u, v = line.split()[:2]
                    g.add_edge(int(u), int(v))
    elif name == "abilene":
        if n != 12:
            raise ValueError("abilene topology needs --nodes 12")
        idx = {s: i for i, s in enumerate(ABILENE_NODES)}
        g.add_edges_from((idx[a], idx[b]) for a, b in ABILENE_LINKS)
    else:
        # generic fallback (e.g. GEANT): ring + deterministic chords.
        # For real GEANT results supply the true link list with --edges.
        rng = np.random.default_rng(1)
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
        for _ in range(n // 2):
            a, b = rng.integers(0, n, 2)
            if a != b:
                g.add_edge(int(a), int(b))
    if not nx.is_connected(g):
        raise ValueError("Topology is not connected")
    return g


def load_link_metadata(g: nx.Graph, path: str):
    """Attach verified per-link capacities and IGP metrics using matrix-order indices."""
    records = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            u, v = sorted((int(row["u_idx"]), int(row["v_idx"])))
            key = (u, v)
            if key in records:
                raise ValueError(f"Duplicate link metadata for {key}")
            capacity = float(row["capacity_kbps"])
            metric = float(row["igp_metric"])
            if capacity <= 0 or metric <= 0:
                raise ValueError(f"Capacity and IGP metric must be positive for {key}")
            records[key] = {"capacity_kbps": capacity, "igp_metric": metric,
                            "source_ids": (row["u_source_id"], row["v_source_id"])}
    graph_edges = {tuple(sorted((int(u), int(v)))) for u, v in g.edges()}
    if set(records) != graph_edges:
        missing = sorted(graph_edges - set(records))
        extra = sorted(set(records) - graph_edges)
        raise ValueError(f"Link metadata does not match topology; missing={missing}, extra={extra}")
    for (u, v), values in records.items():
        g[u][v].update(values)
    return records


class PathTable:
    """k-shortest paths for every OD pair, stored as incidence arrays over directed links."""

    def __init__(self, g: nx.Graph, k: int = 3):
        self.n = g.number_of_nodes()
        self.k = k
        self.edges = [(u, v) for u, v in g.edges()] + [(v, u) for u, v in g.edges()]
        eid = {e: i for i, e in enumerate(self.edges)}
        self.E = len(self.edges)
        self.flows = [(i, j) for i in range(self.n) for j in range(self.n) if i != j]
        F = len(self.flows)
        self.A = np.zeros((F, k, self.E), dtype=np.float32)   # A[f, p, e] = 1 if path p of flow f uses link e
        self.valid = np.zeros((F, k), dtype=bool)
        self.hops = np.zeros((F, k), dtype=np.float32)
        self.costs = np.zeros((F, k), dtype=np.float32)
        weight = "igp_metric" if all("igp_metric" in d for _, _, d in g.edges(data=True)) else None
        for f, (s, t) in enumerate(self.flows):
            paths = list(itertools.islice(nx.shortest_simple_paths(g, s, t, weight=weight), k))
            for p, path in enumerate(paths):
                self.valid[f, p] = True
                self.hops[f, p] = len(path) - 1
                self.costs[f, p] = sum(float(g[a][b].get(weight, 1.0)) if weight else 1.0
                                       for a, b in zip(path[:-1], path[1:]))
                for a, b in zip(path[:-1], path[1:]):
                    self.A[f, p, eid[(a, b)]] = 1.0

    def demand_vector(self, tm: np.ndarray) -> np.ndarray:
        return np.array([tm[s, t] for s, t in self.flows], dtype=np.float32)

    def loads(self, demand, choice):
        return (demand[:, None] * self.A[np.arange(len(choice)), choice]).sum(axis=0)


# ----------------------------------------------------------------------------
# Routing decision
# ----------------------------------------------------------------------------
def _capacity_vector(cap, n_links):
    values = np.asarray(cap, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(n_links, float(values), dtype=np.float32)
    if values.shape != (n_links,) or np.any(values <= 0):
        raise ValueError(f"Capacity must be positive scalar or length-{n_links} vector")
    return values


def optimize(pt: PathTable, demand, cap, thr=0.8, max_moves=100):
    """Greedy congestion relief. Start from shortest paths; while some link is
    above `thr`, move one flow off the worst link to the alternative path that
    lowers the maximum link utilisation the most."""
    F = len(demand)
    cap = _capacity_vector(cap, pt.E)
    choice = np.zeros(F, dtype=int)
    load = pt.loads(demand, choice)
    for _ in range(max_moves):
        util = load / cap
        worst = int(util.argmax())
        if util[worst] <= thr:
            break
        cur_max = util[worst]
        on_worst = np.where((pt.A[np.arange(F), choice, worst] > 0) & (demand > 0))[0]
        best_score, best_move = cur_max - 1e-6, None
        for f in on_worst:
            cur_vec = pt.A[f, choice[f]]
            for p in range(pt.k):
                if p == choice[f] or not pt.valid[f, p]:
                    continue
                new_load = load - demand[f] * cur_vec + demand[f] * pt.A[f, p]
                score = (new_load / cap).max()
                if score < best_score:
                    best_score, best_move = score, (f, p, new_load)
        if best_move is None:
            break
        f, p, load = best_move
        choice[f] = p
    return choice


def optimize_overflow(pt: PathTable, demand, cap, thr=0.8, max_moves=100):
    """Greedily reduce squared predicted overload across all directed links."""
    F = len(demand)
    cap = _capacity_vector(cap, pt.E)
    choice = np.zeros(F, dtype=int)
    load = pt.loads(demand, choice)

    def cost(x):
        excess = np.maximum(x / cap - thr, 0.0)
        return float(excess @ excess)

    current = cost(load)
    for _ in range(max_moves):
        util = load / cap
        if util.max() <= thr:
            break
        worst = int(util.argmax())
        on_worst = np.where((pt.A[np.arange(F), choice, worst] > 0) & (demand > 0))[0]
        best_score, best_move = current - 1e-12, None
        for f in on_worst:
            cur_vec = pt.A[f, choice[f]]
            for p in range(pt.k):
                if p == choice[f] or not pt.valid[f, p]:
                    continue
                new_load = load - demand[f] * cur_vec + demand[f] * pt.A[f, p]
                score = cost(new_load)
                if score < best_score:
                    best_score, best_move = score, (f, p, new_load)
        if best_move is None:
            break
        f, p, load = best_move
        choice[f] = p
        current = best_score
    return choice


def optimize_overflow_persistent(pt: PathTable, demand, cap, thr, previous,
                                 penalty=0.0, max_moves=100, objective="overflow",
                                 max_changes=None):
    """Overflow optimizer that starts from the last route assignment.

    The objective is squared utilization excess above ``thr`` plus a penalty
    for changing flows from ``previous``. ``max_moves`` is a hard upper bound
    on whole-flow path changes in this interval.
    """
    demand = np.maximum(np.asarray(demand, dtype=np.float32), 0)
    cap = _capacity_vector(cap, pt.E)
    previous = np.asarray(previous, dtype=int)
    if previous.shape != (len(demand),):
        raise ValueError("previous route vector must have one path index per flow")
    if penalty < 0:
        raise ValueError("reroute penalty must be non-negative")
    if max_changes is not None and max_changes < 0:
        raise ValueError("max_changes must be non-negative or None")

    choice = previous.copy()
    flow_idx = np.arange(len(demand))
    load = pt.loads(demand, choice)

    if objective not in ("overflow", "minmax"):
        raise ValueError("objective must be 'overflow' or 'minmax'")

    def score(values, routes):
        util = values / cap
        route_cost = penalty * np.count_nonzero(routes != previous)
        if objective == "minmax":
            return float(util.max() + route_cost)
        excess = np.maximum(values / cap - thr, 0.0)
        return float(excess @ excess + route_cost)

    current = score(load, choice)
    for _ in range(max_moves):
        util = load / cap
        if util.max() <= thr:
            break
        worst = int(util.argmax())
        active = np.where((pt.A[flow_idx, choice, worst] > 0) & (demand > 0))[0]
        best_score, best_move = current - 1e-12, None
        for f in active:
            old_path = pt.A[f, choice[f]]
            for p in range(pt.k):
                if p == choice[f] or not pt.valid[f, p]:
                    continue
                candidate_load = load - demand[f] * old_path + demand[f] * pt.A[f, p]
                route_delta = int(p != previous[f]) - int(choice[f] != previous[f])
                candidate_routes = choice.copy()
                candidate_routes[f] = p
                if (max_changes is not None and
                        np.count_nonzero(candidate_routes != previous) > max_changes):
                    continue
                candidate_score = score(candidate_load, candidate_routes)
                if candidate_score < best_score:
                    best_score, best_move = candidate_score, (f, p, candidate_load)
        if best_move is None:
            break
        f, p, load = best_move
        choice[f] = p
        current = best_score
    return choice


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(pt, true, pred, thr, cap_pct, cap_util, lag=1, proactive_thr=None,
             capacity=None, last_observed=None, forecast_label="CNN-LSTM", optimizer="minmax",
             persistent=False, reroute_penalty=0.0, max_moves=100,
             persistent_objective="overflow", max_route_changes=None):
    T = len(true)
    dem_true = [pt.demand_vector(true[t]) for t in range(T)]
    dem_pred = [pt.demand_vector(pred[t]) for t in range(T)]
    if last_observed is None:
        last_observed = np.stack([true[max(t - lag, 0)] for t in range(T)])
    dem_last = [pt.demand_vector(last_observed[t]) for t in range(T)]
    # hybrid: per OD pair, the larger of the forecast and the last matrix the controller has actually seen
    dem_hyb = [pt.demand_vector(np.maximum(pred[t], last_observed[t])) for t in range(T)]
    zero = np.zeros(len(pt.flows), dtype=int)

    # With per-link metadata, capacity is a vector. Otherwise retain the uniform
    # train-prefix calibration behavior used by the stress-test experiment.
    if capacity is None:
        # Legacy fallback for callers without a calibration set. For reported
        # results, pass a train-only calibration series or a fixed capacity.
        static_max = np.array([pt.loads(d, zero).max() for d in dem_true])
        cap = np.percentile(static_max, cap_pct) / cap_util
    else:
        cap = float(capacity) if np.ndim(capacity) == 0 else np.asarray(capacity, dtype=np.float32)

    pthr = thr if proactive_thr is None else proactive_thr     # threshold the proactive controller acts on
    optimize_fn = optimize if optimizer == "minmax" else optimize_overflow

    def controller_routes(demands, threshold):
        previous = zero.copy()
        choices = []
        for demand in demands:
            if persistent:
                choice = optimize_overflow_persistent(
                    pt, demand, cap, threshold, previous, reroute_penalty, max_moves,
                    persistent_objective, max_route_changes)
            else:
                choice = optimize_fn(pt, demand, cap, threshold)
            choices.append(choice)
            previous = choice.copy()
        return choices

    reactive_choices = controller_routes(dem_last, thr)
    proactive_choices = controller_routes(dem_pred, pthr)
    hybrid_choices = controller_routes(dem_hyb, pthr)
    oracle_choices = controller_routes(dem_true, thr)
    schemes = {
        "static": lambda t: zero,
        "reactive (last matrix)": lambda t: reactive_choices[t],
        f"proactive ({forecast_label})": lambda t: proactive_choices[t],
        "hybrid (max fcst, last)": lambda t: hybrid_choices[t],
        "oracle (true matrix)": lambda t: oracle_choices[t],
    }

    out = {}
    for name, route in schemes.items():
        max_util, congested, above_trigger, overload, over_trigger, rerouted, stretch, route_changes = [], [], [], [], [], [], [], []
        previous = zero.copy()
        for t in range(T):
            ch = route(t)
            load = pt.loads(dem_true[t], ch)
            util = load / cap
            max_util.append(util.max())
            congested.append(float((util > 1.0).any()))
            above_trigger.append(float((util > thr).any()))
            total = max(load.sum(), 1e-9)
            overload.append(np.maximum(load - cap, 0).sum() / total)   # proxy for dropped-traffic fraction
            over_trigger.append(np.maximum(load - thr * cap, 0).sum() / total)
            rerouted.append(int((ch != 0).sum()))
            route_changes.append(int((ch != previous).sum()))
            previous = ch.copy()
            h_new = (dem_true[t] * pt.costs[np.arange(len(ch)), ch]).sum()
            h_old = (dem_true[t] * pt.costs[:, 0]).sum()
            stretch.append(h_new / max(h_old, 1e-9))
        out[name] = {
            "mean_max_link_util": float(np.mean(max_util)),
            "p95_max_link_util": float(np.percentile(max_util, 95)),
            "steps_with_congestion_%": float(np.mean(congested) * 100),
            "overload_traffic_%": float(np.mean(overload) * 100),
            "steps_above_trigger_%": float(np.mean(above_trigger) * 100),
            "over_trigger_traffic_proxy_%": float(np.mean(over_trigger) * 100),
            "avg_flows_rerouted": float(np.mean(rerouted)),
            "avg_route_changes_per_step": float(np.mean(route_changes)),
            "path_stretch": float(np.mean(stretch)),
        }
    return out, cap


def detection_metrics(pt, true, views, cap, thr, view_thr=None):
    """How well does each matrix a controller could use detect congested links?

    Link loads come from static shortest-path routing of the matrix. A (step, directed link)
    is 'congested' when load / cap > thr. Each view is scored against the congestion that the
    TRUE matrix produces. link_load_RMSE is in % of link capacity.
    """
    zero = np.zeros(len(pt.flows), dtype=int)

    def loads(arr):
        return np.array([pt.loads(pt.demand_vector(arr[t]), zero) for t in range(len(arr))])

    L_true = loads(true)
    truth = L_true / cap > thr
    out = {}
    for name, arr in views.items():
        L = loads(arr)
        flag = L / cap > (view_thr or {}).get(name, thr)
        tp = int((flag & truth).sum())
        fp = int((flag & ~truth).sum())
        fn = int((~flag & truth).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out[name] = {
            "link_load_RMSE_%cap": float(100 * np.sqrt((((L - L_true) / cap) ** 2).mean())),
            "precision": prec,
            "recall": rec,
            "F1": 2 * prec * rec / max(prec + rec, 1e-9),
            "true_congested_events": int(truth.sum()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="folder with pred_test.npy / true_test.npy from train.py")
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--topo", default="abilene")
    ap.add_argument("--edges", default=None, help="optional edge-list file: one 'u v' pair per line (node indices)")
    ap.add_argument("--calibration", default=None,
                    help="full traffic-matrix file used only for train-period capacity calibration (first 70%%)")
    ap.add_argument("--capacity", type=float, default=None,
                    help="explicit uniform link capacity; overrides --calibration")
    ap.add_argument("--link-metadata", default=None,
                    help="CSV of per-link capacity_kbps and igp_metric, keyed by matrix-order u_idx,v_idx")
    ap.add_argument("--k", type=int, default=3, help="candidate paths per flow")
    ap.add_argument("--thr", type=float, default=0.8, help="predicted-congestion threshold (link utilisation)")
    ap.add_argument("--cap-pct", type=float, default=80, help="percentile used to size link capacity")
    ap.add_argument("--cap-util", type=float, default=0.9, help="static max-utilisation at that percentile")
    ap.add_argument("--proactive-thr", type=float, default=None,
                    help="threshold the PROACTIVE controller acts on (default = --thr). Forecasts trained with squared error shrink peaks, so a value below --thr adds a safety margin")
    ap.add_argument("--optimizer", choices=("minmax", "overflow"), default="minmax",
                    help="greedy objective: lower worst link (minmax) or total squared overload (overflow)")
    ap.add_argument("--persistent", action="store_true",
                    help="warm-start each controller from its own previous routes")
    ap.add_argument("--reroute-penalty", type=float, default=0.0,
                    help="persistent-policy cost per flow moved away from its previous path")
    ap.add_argument("--persistent-objective", choices=("overflow", "minmax"), default="overflow",
                    help="persistent objective: total squared threshold excess or peak utilization")
    ap.add_argument("--max-moves", type=int, default=100,
                    help="maximum whole-flow path changes per interval for --persistent")
    ap.add_argument("--max-route-changes", type=int, default=None,
                    help="optional hard limit on final whole-flow route changes per interval for --persistent")
    ap.add_argument("--margin-sweep", action="store_true",
                    help="rerun reactive / proactive / oracle at several controller thresholds (fair comparison)")
    ap.add_argument("--sweep", action="store_true", help="also print forecast detection quality at several flag thresholds")
    ap.add_argument("--lag", type=int, default=1,
                    help="steps between the last observation and the forecast target (= --horizon used in train.py)")
    ap.add_argument("--forecast-label", default="CNN-LSTM",
                    help="label for pred_test.npy in the saved routing report")
    args = ap.parse_args()
    if args.persistent and args.optimizer != "overflow":
        ap.error("--persistent currently requires --optimizer overflow")
    if args.reroute_penalty < 0 or args.max_moves < 1:
        ap.error("--reroute-penalty must be non-negative and --max-moves must be positive")

    true = np.load(f"{args.results}/true_test.npy")
    pred = np.load(f"{args.results}/pred_test.npy")
    last_path = f"{args.results}/last_observed_test.npy"
    last_observed = np.load(last_path) if os.path.exists(last_path) else None
    g = build_topology(args.topo, args.nodes, args.edges)
    link_metadata = load_link_metadata(g, args.link_metadata) if args.link_metadata else None
    print(f"Topology: {g.number_of_nodes()} nodes, {g.number_of_edges()} links; {len(true)} test steps")
    pt = PathTable(g, args.k)

    capacity = args.capacity
    if link_metadata is not None:
        if capacity is not None or args.calibration:
            raise ValueError("--link-metadata cannot be combined with --capacity or --calibration")
        capacity = np.array([g[u][v]["capacity_kbps"] for u, v in pt.edges], dtype=np.float32)
    if capacity is None and args.calibration:
        calibration, _ = despike(load_tm(args.calibration, args.nodes))
        calibration = calibration[:int(0.7 * len(calibration))]
        zero = np.zeros(len(pt.flows), dtype=int)
        static_max = np.array([pt.loads(pt.demand_vector(tm), zero).max() for tm in calibration])
        capacity = float(np.percentile(static_max, args.cap_pct) / args.cap_util)
    if last_observed is not None and len(last_observed) != len(true):
        raise ValueError("last_observed_test.npy must have the same number of rows as true_test.npy")
    res, cap = evaluate(pt, true, pred, args.thr, args.cap_pct, args.cap_util,
                        args.lag, args.proactive_thr, capacity, last_observed,
                        args.forecast_label, args.optimizer, args.persistent,
                        args.reroute_penalty, args.max_moves,
                        args.persistent_objective, args.max_route_changes)
    if np.ndim(cap) == 0:
        print(f"Uniform link capacity = {cap:.1f} traffic units, congestion threshold = {args.thr:.0%}\n")
    else:
        print(f"Heterogeneous link capacities = {len(cap)} directed links, "
              f"{np.min(cap):.0f} to {np.max(cap):.0f} kbps; congestion threshold = {args.thr:.0%}\n")
    cols = list(next(iter(res.values())).keys())
    print("{:<26}".format("Scheme") + "".join(f"{c[:16]:>18}" for c in cols))
    for name, r in res.items():
        print("{:<26}".format(name) + "".join(f"{r[c]:>18.3f}" for c in cols))

    if last_observed is not None:
        last_seen = last_observed
    else:
        last_seen = np.concatenate([np.repeat(true[:1], args.lag, axis=0), true[:-args.lag]]) if args.lag > 0 else true
    forecast_name = f"{args.forecast_label} forecast"
    det = detection_metrics(pt, true, {f"last observed ({args.lag} step(s) old)": last_seen,
                                       forecast_name: pred,
                                       "hybrid (max fcst, last)": np.maximum(pred, last_seen)}, cap, args.thr,
                            view_thr={forecast_name: args.proactive_thr or args.thr,
                                      "hybrid (max fcst, last)": args.proactive_thr or args.thr})
    dcols = list(next(iter(det.values())).keys())
    print(f"\nCongestion detection on link loads (congested = load > {args.thr:.0%} of capacity)")
    print("{:<34}".format("Matrix used") + "".join(f"{c[:20]:>22}" for c in dcols))
    for name, r in det.items():
        print("{:<34}".format(name) + "".join(f"{r[c]:>22.3f}" for c in dcols))

    if args.sweep:
        print(f"\nForecast flagged at different thresholds (true congestion stays at {args.thr:.0%})")
        for t_ in (0.85, 0.80, 0.75, 0.70):
            r = detection_metrics(pt, true, {forecast_name: pred}, cap, args.thr,
                                  {forecast_name: t_})[forecast_name]
            print(f"  flag at {t_:.0%}: precision {r['precision']:.3f}  recall {r['recall']:.3f}  F1 {r['F1']:.3f}")

    if args.margin_sweep:
        print("\nController threshold sweep: the SAME threshold for reactive, proactive and oracle")
        print("each cell = congested steps %, overload traffic %, avg flows rerouted")
        print("{:<6}{:<30}{:<30}{:<30}{:<30}".format("thr", "reactive", "proactive", "hybrid", "oracle"))
        for t_ in (0.85, 0.80, 0.75, 0.70):
            r_, _ = evaluate(pt, true, pred, t_, args.cap_pct, args.cap_util, args.lag,
                             capacity=capacity, last_observed=last_observed,
                             forecast_label=args.forecast_label, optimizer=args.optimizer,
                             persistent=args.persistent, reroute_penalty=args.reroute_penalty,
                                 max_moves=args.max_moves,
                                 persistent_objective=args.persistent_objective,
                                 max_route_changes=args.max_route_changes)
            cells = []
            for key in ("reactive (last matrix)", f"proactive ({args.forecast_label})",
                        "hybrid (max fcst, last)", "oracle (true matrix)"):
                v = r_[key]
                cells.append(f"{v['steps_with_congestion_%']:6.2f} {v['overload_traffic_%']:7.3f} {v['avg_flows_rerouted']:6.2f}")
            print(f"{t_:<6.0%}" + "".join(f"{c:<30}" for c in cells))

    cap_json = (float(cap) if np.ndim(cap) == 0 else None)
    link_caps_json = ([{"u": int(u), "v": int(v), "capacity_kbps": float(g[u][v]["capacity_kbps"]),
                        "igp_metric": float(g[u][v]["igp_metric"])}
                       for u, v in g.edges()] if link_metadata is not None else None)
    with open(f"{args.results}/routing_results.json", "w") as f:
        json.dump({"metadata": {"topology": args.topo, "edge_file": args.edges,
                                 "k_paths": args.k, "routing_objective": args.optimizer,
                                 "capacity": cap_json, "link_capacities": link_caps_json,
                                 "link_metadata_file": args.link_metadata,
                                 "persistent_routes": args.persistent,
                                 "reroute_penalty": args.reroute_penalty,
                                 "persistent_objective": args.persistent_objective,
                                 "max_route_changes": args.max_route_changes,
                                 "max_moves_per_interval": args.max_moves,
                                 "link_metadata_provenance": (
                                     "IGen geant-20041125.gml; unique graph-isomorphism match to "
                                     "the anonymized 23-node, 37-link topology; bandwidth converted to kbps"
                                     if link_metadata is not None else None),
                                 "capacity_calibration_file": args.calibration,
                                 "congestion_threshold": args.thr,
                                 "proactive_threshold": args.proactive_thr or args.thr,
                                 "forecast_label": args.forecast_label,
                                 "capacity_source": ("explicit" if args.capacity is not None else
                                                     "per-link metadata file" if link_metadata is not None else
                                                     "train-prefix calibration" if args.calibration else
                                                     "test fallback"),
                                 "last_observation_file": last_path if last_observed is not None else None},
                   "routing": res, "detection": det}, f, indent=2)
    print(f"\nSaved {args.results}/routing_results.json")


if __name__ == "__main__":
    main()
