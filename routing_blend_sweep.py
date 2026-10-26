"""Select a forecast/last-observation blend for routing using validation only."""
import json
import numpy as np
from data import load_tm, despike
from routing import build_topology, PathTable, optimize
from routing_validation_sweep import measure


def main():
    n=23; results="res_STACK_BEST"; cap_pct=5.; cap_util=.9; threshold=.7
    true=np.load(f"{results}/true_val.npy")
    pred=np.load(f"{results}/pred_val.npy")
    last=np.load(f"{results}/last_observed_val.npy")
    g=build_topology("geant",n,"geant_edges.txt"); pt=PathTable(g,k=3)
    tm,_=despike(load_tm("geant_final.npy",n)); train=tm[:int(.7*len(tm))]
    zero=np.zeros(len(pt.flows),dtype=int)
    static_max=np.array([pt.loads(pt.demand_vector(x),zero).max() for x in train])
    cap=float(np.percentile(static_max,cap_pct)/cap_util)
    true_demands=[pt.demand_vector(x) for x in true]
    last_demands=[pt.demand_vector(x) for x in last]
    reactive=measure(pt,true_demands,last_demands,cap,threshold,"minmax")
    pred_demands=[pt.demand_vector(x) for x in pred]
    rows=[]
    for weight_last in np.round(np.linspace(0,1,11),2):
        mixed=[(1-weight_last)*p+weight_last*l for p,l in zip(pred_demands,last_demands)]
        score=measure(pt,true_demands,mixed,cap,threshold,"minmax")
        rows.append({"weight_last":float(weight_last),**{
            f"proactive_{k}":v for k,v in score.items()},**{
            f"reactive_{k}":v for k,v in reactive.items()}})
    eligible=[r for r in rows if r["proactive_congestion_%"] < reactive["congestion_%"]
              and r["proactive_overload_%"] < reactive["overload_%"]]
    selected=min(eligible,key=lambda r:(r["proactive_congestion_%"],
                 r["proactive_overload_%"],r["proactive_rerouted"])) if eligible else None
    out={"split":"validation","capacity":cap,"threshold":threshold,
         "selection_rule":"require lower congestion and overload proxy than reactive; then minimize congestion, overload, reroutes",
         "reactive":reactive,"selected":selected,"sweep":rows}
    with open(f"{results}/routing_blend_validation.json","w",encoding="utf-8") as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
