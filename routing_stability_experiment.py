"""Compare independent rerouting with persistent routes and 5% hysteresis."""
import json
import numpy as np
from data import load_tm, despike
from routing import build_topology, PathTable, optimize


def optimize_stable(pt,demand,cap,thr,previous,hysteresis=.05,max_moves=100):
    F=len(demand);choice=previous.copy();load=pt.loads(demand,choice)
    for _ in range(max_moves):
        util=load/cap;worst=int(util.argmax());cur_max=float(util[worst])
        if cur_max<=thr:break
        on=np.where((pt.A[np.arange(F),choice,worst]>0)&(demand>0))[0]
        best_score=cur_max*(1-hysteresis);best=None
        for f in on:
            old=pt.A[f,choice[f]]
            for p in range(pt.k):
                if p==choice[f] or not pt.valid[f,p]:continue
                new_load=load-demand[f]*old+demand[f]*pt.A[f,p]
                score=float((new_load/cap).max())
                if score<best_score:
                    best_score=score;best=(f,p,new_load)
        if best is None:break
        f,p,load=best;choice[f]=p
    return choice


def run(pt,true,predictor,cap,thr,hysteresis=None):
    F=len(pt.flows);zero=np.zeros(F,dtype=int);prev=zero.copy()
    con=[];over=[];maxu=[];rer=[];switches=[]
    for yt,xt in zip(true,predictor):
        dtrue=pt.demand_vector(yt);droute=pt.demand_vector(xt)
        ch=optimize(pt,droute,cap,thr) if hysteresis is None else optimize_stable(pt,droute,cap,thr,prev,hysteresis)
        load=pt.loads(dtrue,ch)
        con.append(float((load>cap).any()));over.append(np.maximum(load-cap,0).sum()/max(load.sum(),1e-9))
        maxu.append(load.max()/cap);rer.append(int((ch!=zero).sum()))
        switches.append(int((ch!=prev).sum()));prev=ch
    return {"congestion_%":100*float(np.mean(con)),"overload_%":100*float(np.mean(over)),
            "mean_max_util":float(np.mean(maxu)),"rerouted_flows":float(np.mean(rer)),
            "path_changes_per_interval":float(np.mean(switches))}


def main():
    n=23;results="res_STACK_BEST";g=build_topology("geant",n,"geant_edges.txt");pt=PathTable(g,k=3)
    tm,_=despike(load_tm("geant_final.npy",n));train=tm[:int(.7*len(tm))]
    z=np.zeros(len(pt.flows),dtype=int)
    cap=float(np.percentile([pt.loads(pt.demand_vector(x),z).max() for x in train],5)/.9)
    report={"capacity":cap,"hysteresis":.05,"split_metrics":{}}
    for split in ("val","test"):
        true=np.load(f"{results}/true_{split}.npy");pred=np.load(f"{results}/pred_{split}.npy")
        last=np.load(f"{results}/last_observed_{split}.npy")
        report["split_metrics"][split]={}
        for label,demand in (("reactive",last),("proactive",pred)):
            report["split_metrics"][split][label]={
                "independent":run(pt,true,demand,cap,.7,None),
                "persistent_hysteresis_5pct":run(pt,true,demand,cap,.7,.05)}
    with open(f"{results}/routing_stability_results.json","w",encoding="utf-8") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
