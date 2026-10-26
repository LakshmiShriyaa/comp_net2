"""Validation-CV subset search for the per-OD forecast stack."""
import json
import os
from itertools import combinations
import numpy as np
from data import load_tm, despike, metrics, norm_metrics
from meta_stack import fit_ridge_stack, apply_stack, score_error


def main():
    n = 23
    cnn_v = [np.load(f"res_S4L{z}/pred_val_all.npy")[:, -1]
             for z in ("", "_s1", "_s2")]
    norm_v = np.load("res_NORM_CNN/pred_val_all.npy")[:, -1]
    ridge_v = np.load("res_PERFLOW_MEAN/pred_val.npy")
    seasonal_v = np.load("res_LINEAR_SEASONAL/pred_val.npy")
    log_v = np.load("res_LOG_FLOW/pred_val.npy")
    tree_v = np.mean([np.load("res_TREE_RANGE/pred_val.npy"), np.load("res_TREE_FULL/pred_val.npy")], 0)
    ma_v = np.load("res_S4L/history_val.npy")[:, -3:].mean(1)
    roll_v = np.load("res_ROLL_CNN/pred_val.npy")
    context_v = np.load("res_FLOW_CONTEXT/pred_val.npy")
    cnn_t = [np.load(f"res_S4L{z}/pred_test_all.npy")[:, -1]
             for z in ("", "_s1", "_s2")]
    norm_t = np.load("res_NORM_CNN/pred_test_all.npy")[:, -1]
    ridge_t = np.load("res_PERFLOW_MEAN/pred_test.npy")
    seasonal_t = np.load("res_LINEAR_SEASONAL/pred_test.npy")
    log_t = np.load("res_LOG_FLOW/pred_test.npy")
    tree_t = np.mean([np.load("res_TREE_RANGE/pred_test.npy"), np.load("res_TREE_FULL/pred_test.npy")], 0)
    ma_t = np.load("res_S4L/history_test.npy")[:, -3:].mean(1)
    roll_t = np.load("res_ROLL_CNN/pred_test.npy")
    context_t = np.load("res_FLOW_CONTEXT/pred_test.npy")
    base_v = np.load("res_PERFLOW_MEAN/last_observed_val.npy")
    base_t = np.load("res_PERFLOW_MEAN/last_observed_test.npy")
    yv = np.load("res_PERFLOW_MEAN/true_val.npy")
    yt = np.load("res_PERFLOW_MEAN/true_test.npy")
    names = ["cnn_seed0", "cnn_seed1", "cnn_seed2", "normalized_loss_cnn", "ridge",
             "seasonal_ridge", "log_ridge", "tree_ensemble", "moving_average", "recursive_cnn"]
    pv_all = [*cnn_v, norm_v, ridge_v, seasonal_v, log_v, tree_v, ma_v, roll_v]
    pt_all = [*cnn_t, norm_t, ridge_t, seasonal_t, log_t, tree_t, ma_t, roll_t]
    mask = ~np.eye(n, dtype=bool)
    tm, _ = despike(load_tm("geant_final.npy", n)); train = tm[:int(len(tm)*.7)]
    lo, hi = train.min(0), train.max(0); scale = np.maximum(hi-lo,1).reshape(-1)
    xv_all = np.stack([(p-base_v)[:,mask]/scale[mask.ravel()] for p in pv_all],axis=1)
    xt_all = np.stack([(p-base_t)[:,mask]/scale[mask.ravel()] for p in pt_all],axis=1)
    yd = ((yv-base_v)[:,mask]/scale[mask.ravel()]).astype(float)
    folds = [(800,1200),(1200,len(yv))]
    alphas=(0.01,0.1,1,10)
    subsets = [tuple(i for i in range(10) if i != j) for j in range(10)]
    subsets += [(0,1,2,4,7,8),(3,4,6,7),(4,5,6,8),
                (0,1,2,3,4,6,7,8),(0,1,2,3,4,5,6,7,9)]
    outputs=[]
    for subset in subsets:
        x=xv_all[:,list(subset),:]
        scores={a:[] for a in alphas}
        for start,end in folds:
            for alpha in alphas:
                beta,bias=fit_ridge_stack(x[:start],yd[:start],alpha)
                scores[alpha].append(score_error(apply_stack(x[start:end],beta,bias),yd[start:end]))
        alpha=min(alphas,key=lambda a:np.mean(scores[a]))
        cv=float(np.mean(scores[alpha]))
        outputs.append((cv,subset,alpha))
        print(f"subset={[names[i] for i in subset]} alpha={alpha} cv_norm_rmse={cv:.6f}")
    best_cv=min(r[0] for r in outputs)
    # Prefer the smallest model within 0.1% of the best CV score to limit variance.
    eligible=[r for r in outputs if r[0] <= best_cv*1.001]
    cv,subset,alpha=min(eligible,key=lambda r:(len(r[1]),r[0]))
    xval=xv_all[:,list(subset),:]; xtest=xt_all[:,list(subset),:]
    beta,bias=fit_ridge_stack(xval,yd,alpha)
    val_delta=apply_stack(xval,beta,bias)
    test_delta=apply_stack(xtest,beta,bias)
    pred_v=base_v.copy(); pred_t=base_t.copy()
    pred_v[:,mask]+=val_delta*scale[mask.ravel()]
    pred_t[:,mask]+=test_delta*scale[mask.ravel()]
    pred_v=np.maximum(pred_v,0); pred_t=np.maximum(pred_t,0)
    for p in (pred_v,pred_t): p[:,np.arange(n),np.arange(n)]=0
    report={"selection":{"strategy":"per-OD ridge stack selected by expanding validation CV",
              "selected_features":[names[i] for i in subset],"alpha":alpha,
              "validation_cv_normalized_rmse":cv,"best_cv_normalized_rmse":best_cv,
              "tie_rule":"smallest candidate within 0.1% of best CV score"},
            "validation":{"persistence":{"original_units":metrics(base_v,yv,mask),"normalized":norm_metrics(base_v,yv,lo,hi,mask)},
                          "stack":{"original_units":metrics(pred_v,yv,mask),"normalized":norm_metrics(pred_v,yv,lo,hi,mask)}},
            "test":{"persistence":{"original_units":metrics(base_t,yt,mask),"normalized":norm_metrics(base_t,yt,lo,hi,mask)},
                    "stack":{"original_units":metrics(pred_t,yt,mask),"normalized":norm_metrics(pred_t,yt,lo,hi,mask)}}}
    out="res_STACK_SUBSET"; os.makedirs(out,exist_ok=True)
    for name,p in (("pred_val",pred_v),("true_val",yv),("last_observed_val",base_v),
                   ("pred_test",pred_t),("true_test",yt),("last_observed_test",base_t)):
        np.save(f"{out}/{name}.npy",p)
    np.savez_compressed(f"{out}/weights.npz",beta=beta,intercept=bias)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
