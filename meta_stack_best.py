"""Reproduce the current validation-CV selected GEANT forecast stack."""
import json
import os
import numpy as np
from data import load_tm, despike, metrics, norm_metrics
from meta_stack import fit_ridge_stack, apply_stack, score_error


def main():
    n=23
    def last(path):
        p=np.load(path); return p[:,-1] if p.ndim==4 else p
    # Candidate forecasts; all model weights were fitted without test targets.
    cnn_v=np.mean([last(f"res_S4L{z}/pred_val_all.npy") for z in ("","_s1","_s2")],0)
    cnn_t=np.mean([last(f"res_S4L{z}/pred_test_all.npy") for z in ("","_s1","_s2")],0)
    norm_v=last("res_NORM_CNN/pred_val_all.npy"); norm_t=last("res_NORM_CNN/pred_test_all.npy")
    ridge_v=np.load("res_PERFLOW_MEAN/pred_val.npy"); ridge_t=np.load("res_PERFLOW_MEAN/pred_test.npy")
    seas_v=np.load("res_LINEAR_SEASONAL/pred_val.npy"); seas_t=np.load("res_LINEAR_SEASONAL/pred_test.npy")
    log_v=np.load("res_LOG_FLOW/pred_val.npy"); log_t=np.load("res_LOG_FLOW/pred_test.npy")
    tree_v=np.mean([np.load("res_TREE_RANGE/pred_val.npy"),np.load("res_TREE_FULL/pred_val.npy")],0)
    tree_t=np.mean([np.load("res_TREE_RANGE/pred_test.npy"),np.load("res_TREE_FULL/pred_test.npy")],0)
    hist_v=np.load("res_S4L/history_val.npy"); hist_t=np.load("res_S4L/history_test.npy")
    ma_v=hist_v[:,-3:].mean(1); ma_t=hist_t[:,-3:].mean(1)
    lstm_v=last("res_SHARED_LSTM_S8/pred_val_all.npy")
    lstm_t=last("res_SHARED_LSTM_S8/pred_test_all.npy")
    pv=[cnn_v,norm_v,ridge_v,seas_v,log_v,tree_v,ma_v,lstm_v]
    pt=[cnn_t,norm_t,ridge_t,seas_t,log_t,tree_t,ma_t,lstm_t]
    names=["cnn_seed_ensemble","normalized_loss_cnn","perflow_mean_ridge","seasonal_ridge",
           "log_flow","tree_ensemble","moving_average_3","shared_lstm"]
    base_v=np.load("res_PERFLOW_MEAN/last_observed_val.npy")
    base_t=np.load("res_PERFLOW_MEAN/last_observed_test.npy")
    yv=np.load("res_PERFLOW_MEAN/true_val.npy"); yt=np.load("res_PERFLOW_MEAN/true_test.npy")
    mask=~np.eye(n,dtype=bool)
    tm,_=despike(load_tm("geant_final.npy",n)); train=tm[:int(len(tm)*.7)]
    lo,hi=train.min(0),train.max(0); scale=np.maximum(hi-lo,1).reshape(-1)
    xv=np.stack([(p-base_v)[:,mask]/scale[mask.ravel()] for p in pv],axis=1)
    xt=np.stack([(p-base_t)[:,mask]/scale[mask.ravel()] for p in pt],axis=1)
    yd=((yv-base_v)[:,mask]/scale[mask.ravel()]).astype(float)
    alphas=(.0001,.001,.003,.01,.03,.1,.3,1,3,10,100)
    gains=(.8,.85,.875,.9,.925,.95,.975,1.0,1.025,1.05,1.1)
    folds=[(800,1200),(1200,len(yv))]
    cv={(a,g):[] for a in alphas for g in gains}; baseline=[]
    for start,end in folds:
        baseline.append(score_error(np.zeros_like(yd[start:end]),yd[start:end]))
        for a in alphas:
            b,c=fit_ridge_stack(xv[:start],yd[:start],a)
            raw=apply_stack(xv[start:end],b,c)
            base_norm=base_v[start:end][:,mask]/scale[mask.ravel()]
            for g in gains:
                clipped_delta=np.maximum(base_norm+g*raw,0)-base_norm
                cv[(a,g)].append(score_error(clipped_delta,yd[start:end]))
    alpha,gain=min(cv,key=lambda k:np.mean(cv[k]))
    cv_score=float(np.mean(cv[(alpha,gain)])); base_cv=float(np.mean(baseline))
    beta,bias=fit_ridge_stack(xv,yd,alpha)
    dv=gain*apply_stack(xv,beta,bias); dt=gain*apply_stack(xt,beta,bias)
    pred_v=base_v.copy(); pred_t=base_t.copy()
    pred_v[:,mask]+=dv*scale[mask.ravel()]; pred_t[:,mask]+=dt*scale[mask.ravel()]
    pred_v=np.maximum(pred_v,0); pred_t=np.maximum(pred_t,0)
    for p in (pred_v,pred_t):p[:,np.arange(n),np.arange(n)]=0
    report={"selection":{"strategy":"per-OD ridge meta-stack; penalty and residual gain selected by expanding chronological validation CV",
              "features":names,"alpha":alpha,"residual_gain":gain,"folds":folds,
              "validation_cv_normalized_rmse":{"persistence":base_cv,"stack":cv_score,
                "improvement_vs_persistence_percent":100*(1-cv_score/base_cv)}},
            "validation":{"persistence":{"original_units":metrics(base_v,yv,mask),"normalized":norm_metrics(base_v,yv,lo,hi,mask)},
                          "stack":{"original_units":metrics(pred_v,yv,mask),"normalized":norm_metrics(pred_v,yv,lo,hi,mask)}},
            "test":{"persistence":{"original_units":metrics(base_t,yt,mask),"normalized":norm_metrics(base_t,yt,lo,hi,mask)},
                    "stack":{"original_units":metrics(pred_t,yt,mask),"normalized":norm_metrics(pred_t,yt,lo,hi,mask)}}}
    out="res_STACK_BEST"; os.makedirs(out,exist_ok=True)
    for name,p in (("pred_val",pred_v),("true_val",yv),("last_observed_val",base_v),
                   ("pred_test",pred_t),("true_test",yt),("last_observed_test",base_t)):
        np.save(f"{out}/{name}.npy",p)
    np.savez_compressed(f"{out}/weights.npz",beta=beta,intercept=bias,residual_gain=gain)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
