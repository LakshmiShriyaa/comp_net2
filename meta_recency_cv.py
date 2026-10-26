"""Compare recency-weighted meta-stacking using chronological validation only."""
import json
import numpy as np
from data import load_tm, despike
from meta_stack import score_error


def fit_weighted(x, y, weights, alpha):
    # One small weighted ridge regression per OD pair, with weighted centering.
    p, f = x.shape[2], x.shape[1]
    b = np.zeros((p, f)); c = np.zeros(p)
    eye = np.eye(f)
    for q in range(p):
        xx, yy = x[:, :, q], y[:, q]
        sw = weights.sum()
        xm = (weights[:, None] * xx).sum(0) / sw
        ym = (weights * yy).sum() / sw
        xc, yc = xx-xm, yy-ym
        b[q] = np.linalg.solve(xc.T @ (weights[:, None]*xc) + alpha*eye,
                               xc.T @ (weights*yc))
        c[q] = ym - xm @ b[q]
    return b, c


def main():
    n=23; mask=~np.eye(n,dtype=bool)
    def last(path):
        p=np.load(path); return p[:,-1] if p.ndim==4 else p
    pv=[np.mean([last(f"res_S4L{s}/pred_val_all.npy") for s in ("","_s1","_s2")],0),
        last("res_NORM_CNN/pred_val_all.npy"), np.load("res_PERFLOW_MEAN/pred_val.npy"),
        np.load("res_LINEAR_SEASONAL/pred_val.npy"), np.load("res_LOG_FLOW/pred_val.npy"),
        np.mean([np.load(f"res_TREE_{s}/pred_val.npy") for s in ("RANGE","FULL")],0),
        np.load("res_S4L/history_val.npy")[:,-3:].mean(1),
        np.load("res_ROLL_CNN/pred_val.npy"), np.load("res_FLOW_CONTEXT/pred_val.npy")]
    base=np.load("res_PERFLOW_MEAN/last_observed_val.npy"); yv=np.load("res_PERFLOW_MEAN/true_val.npy")
    tm,_=despike(load_tm("geant_final.npy",n)); train=tm[:int(len(tm)*.7)]
    scale=np.maximum(train.max(0)-train.min(0),1).reshape(-1)[mask.ravel()]
    x=np.stack([(p-base)[:,mask]/scale for p in pv],axis=1)
    y=((yv-base)[:,mask]/scale).astype(float)
    folds=[(800,1200),(1200,len(y))]
    configs={"uniform":None}
    for length in (200,400,600,800,1200): configs[f"window_{length}"]=length
    for half in (100,200,400,800,1600): configs[f"half_life_{half}"]=half
    alphas=(.001,.01,.1,1,10,100)
    persistence=float(np.mean([score_error(np.zeros_like(y[s:e]),y[s:e]) for s,e in folds]))
    results={}
    for label,setting in configs.items():
        all_scores={a:[] for a in alphas}
        for start,end in folds:
            age=np.arange(start-1,-1,-1,dtype=float)
            if setting is None: weights=np.ones(start)
            elif label.startswith("window"): weights=(age < setting).astype(float)
            else: weights=np.exp(-np.log(2)*age/setting)
            for a in alphas:
                b,c=fit_weighted(x[:start],y[:start],weights,a)
                pred=np.einsum("sfq,qf->sq",x[start:end],b)+c
                all_scores[a].append(score_error(pred,y[start:end]))
        alpha=min(alphas,key=lambda a:np.mean(all_scores[a]))
        score=float(np.mean(all_scores[alpha]))
        results[label]={"alpha":alpha,"fold_scores":all_scores[alpha],"mean":score,
                        "improvement_vs_persistence_percent":100*(1-score/persistence)}
    results["persistence_cv_rmse"]=persistence
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
