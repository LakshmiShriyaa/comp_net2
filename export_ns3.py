"""Export controller decisions for the NS-3 simulation.

Writes two CSV files that ns3_wan_sim.cc reads:
    demands.csv       step,src,dst,mbps      traffic to inject in each time step (from the TRUE matrices)
    weights_<mode>.csv step,link,weight      per-link routing weight to apply at the start of each step

Modes
    static     : weight 1 on every link (no controller)
    proactive  : weights from the CNN-LSTM PREDICTED matrix. Links predicted to exceed
                 the threshold get a higher weight, so shortest-path routing steers
                 traffic away from them BEFORE the congestion happens.
    reactive   : same rule, but driven by the previous step's observed matrix.

Usage
    python export_ns3.py --results results --steps 40 --link-mbps 100 --out ns3_input
"""
import argparse
import os
import numpy as np

from routing import build_topology, PathTable, ABILENE_NODES, ABILENE_LINKS


def link_utils(pt, tm, idx_of_link, cap):
    """Static-shortest-path utilisation of each undirected Abilene link for matrix tm."""
    load = pt.loads(pt.demand_vector(tm), np.zeros(len(pt.flows), dtype=int))
    util = np.zeros(len(ABILENE_LINKS))
    for e, (u, v) in enumerate(pt.edges):
        li = idx_of_link[frozenset((u, v))]
        util[li] = max(util[li], load[e] / cap)
    return util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="ns3_input")
    ap.add_argument("--steps", type=int, default=40, help="number of test steps to simulate")
    ap.add_argument("--start", type=int, default=0, help="first test step")
    ap.add_argument("--link-mbps", type=float, default=100.0, help="link rate used in NS-3")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--penalty", type=float, default=10.0, help="extra weight per 100%% of utilisation above thr")
    ap.add_argument("--cap-pct", type=float, default=70)
    ap.add_argument("--cap-util", type=float, default=1.0)
    args = ap.parse_args()

    true = np.load(f"{args.results}/true_test.npy")
    pred = np.load(f"{args.results}/pred_test.npy")
    g = build_topology("abilene", 12)
    pt = PathTable(g, k=1)
    name_idx = {s: i for i, s in enumerate(ABILENE_NODES)}
    idx_of_link = {frozenset((name_idx[a], name_idx[b])): i for i, (a, b) in enumerate(ABILENE_LINKS)}

    # same capacity rule as routing.py, so both evaluations see the same congestion level
    zero = np.zeros(len(pt.flows), dtype=int)
    static_max = np.array([pt.loads(pt.demand_vector(m), zero).max() for m in true])
    cap = np.percentile(static_max, args.cap_pct) / args.cap_util
    mbps_per_unit = args.link_mbps / cap          # traffic units -> Mbps so capacity maps to link-mbps

    os.makedirs(args.out, exist_ok=True)
    steps = range(args.start, min(args.start + args.steps, len(true)))

    with open(f"{args.out}/demands.csv", "w") as f:
        f.write("step,src,dst,mbps\n")
        for s, t in enumerate(steps):
            for i in range(12):
                for j in range(12):
                    if i != j and true[t, i, j] > 0:
                        f.write(f"{s},{i},{j},{true[t, i, j] * mbps_per_unit:.4f}\n")

    def write_weights(mode):
        with open(f"{args.out}/weights_{mode}.csv", "w") as f:
            f.write("step,link,weight\n")
            for s, t in enumerate(steps):
                if mode == "static":
                    util = np.zeros(len(ABILENE_LINKS))
                elif mode == "proactive":
                    util = link_utils(pt, pred[t], idx_of_link, cap)
                else:  # reactive: previous observed matrix
                    util = link_utils(pt, true[max(t - 1, 0)], idx_of_link, cap)
                for li in range(len(ABILENE_LINKS)):
                    w = 1 + max(0.0, util[li] - args.thr) * args.penalty
                    f.write(f"{s},{li},{max(1, int(round(w)))}\n")

    for mode in ("static", "reactive", "proactive"):
        write_weights(mode)
    print(f"Wrote demands.csv and weights_{{static,reactive,proactive}}.csv for {len(steps)} steps to {args.out}/")
    print(f"Link rate {args.link_mbps} Mbps; 1 traffic unit = {mbps_per_unit:.5f} Mbps")


if __name__ == "__main__":
    main()
