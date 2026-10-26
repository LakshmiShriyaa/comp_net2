"""Tune the proactive congestion trigger on validation only; never read test labels."""
import json
import numpy as np
from data import load_tm, despike
from routing import build_topology, PathTable, optimize, optimize_overflow


def measure(pt, true_demands, route_demands, cap, threshold, objective="minmax"):
    congested=[]; overload=[]; rerouted=[]; stretch=[]
    zero=np.zeros(len(pt.flows),dtype=int)
    optimizer=optimize if objective=="minmax" else optimize_overflow
    for demand_true, demand_route in zip(true_demands,route_demands):
        choice=optimizer(pt,demand_route,cap,threshold)
        load=pt.loads(demand_true,choice)
        congested.append(float((load>cap).any()))
        overload.append(np.maximum(load-cap,0).sum()/max(load.sum(),1e-9))
        rerouted.append(int((choice!=zero).sum()))
        new_hops=(demand_true*pt.hops[np.arange(len(choice)),choice]).sum()
        old_hops=(demand_true*pt.hops[:,0]).sum()
        stretch.append(new_hops/max(old_hops,1e-9))
    return {"congestion_%":100*float(np.mean(congested)),
            "overload_%":100*float(np.mean(overload)),
            "rerouted":float(np.mean(rerouted)),
            "path_stretch":float(np.mean(stretch))}


def main():
    n=23; results="res_STACK_BEST"; cap_pct=5.; cap_util=.9; reactive_thr=.7
    true=np.load(f"{results}/true_val.npy")
    pred=np.load(f"{results}/pred_val.npy")
    last=np.load(f"{results}/last_observed_val.npy")
    g=build_topology("geant",n,"geant_edges.txt")
    tm,_=despike(load_tm("geant_final.npy",n)); train=tm[:int(.7*len(tm))]
    thresholds=(.60,.65,.70)
    rows=[]
    reactive_by_objective={}
    for objective in ("minmax","overflow"):
      for k in (3,):
        pt=PathTable(g,k=k)
        zero=np.zeros(len(pt.flows),dtype=int)
        static_max=np.array([pt.loads(pt.demand_vector(x),zero).max() for x in train])
        cap=float(np.percentile(static_max,cap_pct)/cap_util)
        true_demands=[pt.demand_vector(x) for x in true]
        pred_demands=[pt.demand_vector(x) for x in pred]
        last_demands=[pt.demand_vector(x) for x in last]
        reactive=measure(pt,true_demands,last_demands,cap,reactive_thr,objective)
        reactive_by_objective[objective]=reactive
        for thrp in thresholds:
            proactive=measure(pt,true_demands,pred_demands,cap,float(thrp),objective)
            rows.append({"objective":objective,"k_paths":k,"proactive_threshold":float(thrp),**{
                         f"proactive_{name}":v for name,v in proactive.items()},
                         **{f"reactive_{name}":v for name,v in reactive.items()}})
    # Prefer Pareto improvements; among them minimize congestion, then overload.
    eligible=[r for r in rows if r["proactive_congestion_%"] < reactive_by_objective[r["objective"]]["congestion_%"]
              and r["proactive_overload_%"] < reactive_by_objective[r["objective"]]["overload_%"]]
    selected=min(eligible,key=lambda r:(r["proactive_congestion_%"],r["proactive_overload_%"],r["proactive_rerouted"])) if eligible else None
    out={"split":"validation","capacity_calibrated_from":"train prefix",
         "reactive_threshold":reactive_thr,"reactive_by_objective":reactive_by_objective,
         "selection_rule":"among thresholds improving both congestion and overload vs reactive, minimize congestion, then overload, then reroutes",
         "reactive_validation":reactive,"selected":selected,"sweep":rows}
    with open(f"{results}/routing_threshold_validation.json","w",encoding="utf-8") as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
