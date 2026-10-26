"""Research prototype: path-splitting min-max routing with validation/test reports."""
import argparse
import json
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from data import load_tm, despike
from routing import build_topology, PathTable


def optimize_split(pt, demand, cap, path_penalty=.01, threshold=.7):
    """Minimize max utilization after a trigger, with fractional flow over k paths."""
    F,K,E=pt.A.shape
    demand=np.maximum(np.asarray(demand,dtype=float),0)
    if pt.loads(demand,np.zeros(F,dtype=int)).max()/cap <= threshold:
        out=np.zeros((F,K));out[:,0]=1.0
        return out
    total=max(float(demand.sum()),1e-12)
    nv=F*K+1; ucol=F*K
    c=np.zeros(nv); c[ucol]=1.0
    shortest=pt.hops[:,0]
    for f in range(F):
        for p in range(K):
            c[f*K+p]=path_penalty*demand[f]/total*max(float(pt.hops[f,p]-shortest[f]),0)
    eq_rows=np.repeat(np.arange(F),K)
    eq_cols=np.arange(F*K)
    Aeq=coo_matrix((np.ones(F*K),(eq_rows,eq_cols)),shape=(F,nv)).tocsr()
    A=np.asarray(pt.A)
    fs,ps,es=np.nonzero(A)
    vals=demand[fs]*A[fs,ps,es]
    Aub=coo_matrix((np.r_[vals,np.full(E,-cap)],
                    (np.r_[es,np.arange(E)],np.r_[fs*K+ps,np.full(E,ucol)])),
                   shape=(E,nv)).tocsr()
    bounds=[(0.0,1.0 if pt.valid[f,p] else 0.0) for f in range(F) for p in range(K)]
    bounds.append((0.0,None))
    res=linprog(c,A_ub=Aub,b_ub=np.zeros(E),A_eq=Aeq,b_eq=np.ones(F),
                bounds=bounds,method="highs")
    if not res.success:
        raise RuntimeError(f"routing LP failed: {res.message}")
    return res.x[:F*K].reshape(F,K)


def summarize(pt,true_demands,fractions_by_t,cap):
    congested=[]; overload=[]; maxutil=[]; rerouted=[]; stretches=[]
    for d,x in zip(true_demands,fractions_by_t):
        load=np.einsum("f,fp,fpe->e",d,x,pt.A)
        congested.append(float((load>cap).any()))
        overload.append(np.maximum(load-cap,0).sum()/max(load.sum(),1e-9))
        maxutil.append(load.max()/cap)
        rerouted.append(float(np.sum((x[:,1:]>1e-5).any(axis=1))))
        hnew=np.sum(d[:,None]*x*pt.hops)
        hold=np.sum(d*pt.hops[:,0])
        stretches.append(hnew/max(hold,1e-9))
    return {"congestion_%":100*float(np.mean(congested)),
            "overload_%":100*float(np.mean(overload)),
            "mean_max_util":float(np.mean(maxutil)),
            "rerouted_flows":float(np.mean(rerouted)),
            "path_stretch":float(np.mean(stretches))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=0,help="0 evaluates full split; otherwise evaluate first N samples")
    ap.add_argument("--k",type=int,default=3)
    ap.add_argument("--path-penalty",type=float,default=.01)
    ap.add_argument("--split",choices=("val","test"),default="val")
    ap.add_argument("--threshold",type=float,default=.7)
    args=ap.parse_args()
    n=23; results="res_STACK_BEST"; cap_pct=5.;cap_util=.9
    g=build_topology("geant",n,"geant_edges.txt");pt=PathTable(g,k=args.k)
    tm,_=despike(load_tm("geant_final.npy",n));train=tm[:int(.7*len(tm))]
    zero=np.zeros(len(pt.flows),dtype=int)
    static_max=np.array([pt.loads(pt.demand_vector(x),zero).max() for x in train])
    cap=float(np.percentile(static_max,cap_pct)/cap_util)
    report={"capacity":cap,"capacity_source":"train-only percentile stress assumption",
            "k_paths":args.k,"path_penalty":args.path_penalty,"threshold":args.threshold,
            "limit":args.limit,"split":args.split}
    for split in (args.split,):
        true=np.load(f"{results}/true_{split}.npy")
        pred=np.load(f"{results}/pred_{split}.npy")
        last=np.load(f"{results}/last_observed_{split}.npy")
        if args.limit:
            true,pred,last=true[:args.limit],pred[:args.limit],last[:args.limit]
        true_d=[pt.demand_vector(x) for x in true]
        pred_d=[pt.demand_vector(x) for x in pred]
        last_d=[pt.demand_vector(x) for x in last]
        forecast_routes=[optimize_split(pt,d,cap,args.path_penalty,args.threshold) for d in pred_d]
        reactive_routes=[optimize_split(pt,d,cap,args.path_penalty,args.threshold) for d in last_d]
        report[split]={"proactive_forecast":summarize(pt,true_d,forecast_routes,cap),
                       "reactive_last_observation":summarize(pt,true_d,reactive_routes,cap)}
    out=f"{results}/routing_mcf_{args.split}.json"
    with open(out,"w",encoding="utf-8") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2));print(f"Saved {out}")

if __name__=="__main__":main()
