"""Histogram gradient boosting with per-flow history and topology context."""
import json
import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from data import load_tm, despike, prepare, metrics, norm_metrics
from flow_context import route_incidence, context_features


def make_features(X, ctx, scale):
    own=X/scale[None,None]
    context=ctx.reshape(len(X),-1,*X.shape[-2:])
    blocks=np.concatenate([own,context],axis=1)
    return blocks.transpose(2,3,0,1).astype(np.float32)


def main():
    n,w,h,step=23,12,4,15
    tm,_=despike(load_tm("geant_final.npy",n)); times=np.load("geant_times.npy")
    d=prepare(tm,w,h,mask_diag=True,times=times,step_min=step,season=96); sc=d["scaler"]
    X={k:sc.inverse(d[k][0]) for k in ("train","val","test")}
    Y={k:sc.inverse(d[k][1][:,-1]) for k in ("train","val","test")}
    tr=d["tm"][:d["t_end"]]; scale=np.maximum(tr.max(0)-tr.min(0),1)
    rows=np.maximum(tr.sum(axis=2).mean(0),1); cols=np.maximum(tr.sum(axis=1).mean(0),1)
    A,paths=route_incidence(n,"geant_edges.txt"); link=tr.reshape(len(tr),n*n)@A
    ps0=np.ones(n*n); ps1=np.ones(n*n)
    for q,path in enumerate(paths):
        if path:
            ps0[q]=max(float(link[:,path].mean()),1)
            ps1[q]=max(float(link[:,path].max(axis=1).mean()),1)
    ps=(ps0.reshape(n,n),ps1.reshape(n,n))
    ctx={k:context_features(X[k],A,paths,rows,cols,ps,n)[:,-3:] for k in X}
    F={k:make_features(X[k],ctx[k],scale) for k in X}
    mask=~np.eye(n,dtype=bool); lo,hi=tr.min(0),tr.max(0)
    Xtr,Ytr,Ftr=X["train"],Y["train"],F["train"]
    stride=4; Xfit=Xtr[::stride]; Yfit=Ytr[::stride]; Ffit=Ftr[:,:,::stride]
    cfgs=[{"max_iter":40,"max_leaf_nodes":7,"min_samples_leaf":40,"l2_regularization":1.0},
          {"max_iter":80,"max_leaf_nodes":7,"min_samples_leaf":40,"l2_regularization":1.0},
          {"max_iter":80,"max_leaf_nodes":15,"min_samples_leaf":40,"l2_regularization":1.0}]
    candidates=[]
    for cfg in cfgs:
        pv=np.zeros_like(Y["val"],dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i==j: continue
                s=scale[i,j]
                y=(Yfit[:,i,j]-Xfit[:,-1,i,j])/s
                model=HistGradientBoostingRegressor(learning_rate=.08,early_stopping=False,random_state=0,**cfg)
                model.fit(Ffit[i,j],y)
                pv[:,i,j]=np.maximum(X["val"][:,-1,i,j]+model.predict(F["val"][i,j])*s,0)
        score=norm_metrics(pv,Y["val"],lo,hi,mask)["RMSE"]
        candidates.append((score,cfg,pv)); print(f"validation config={cfg} normalized_rmse={score:.6f}")
    best,cfg,pv=min(candidates,key=lambda z:z[0])
    Xall=np.concatenate([X["train"],X["val"]]); Yall=np.concatenate([Y["train"],Y["val"]]); Fall=np.concatenate([F["train"],F["val"]],axis=2)
    pt=np.zeros_like(Y["test"],dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            s=scale[i,j]
            model=HistGradientBoostingRegressor(learning_rate=.08,early_stopping=False,random_state=0,**cfg)
            model.fit(Fall[i,j],(Yall[:,i,j]-Xall[:,-1,i,j])/s)
            pt[:,i,j]=np.maximum(X["test"][:,-1,i,j]+model.predict(F["test"][i,j])*s,0)
    last_v,last_t=X["val"][:,-1].copy(),X["test"][:,-1].copy()
    for p in (pv,pt,last_v,last_t): p[:,np.arange(n),np.arange(n)]=0
    report={"selection":{"config":cfg,"validation_normalized_rmse":best,"train_stride":stride},
            "validation":{"persistence":{"original_units":metrics(last_v,Y["val"],mask),"normalized":norm_metrics(last_v,Y["val"],lo,hi,mask)},
                          "tree_context":{"original_units":metrics(pv,Y["val"],mask),"normalized":norm_metrics(pv,Y["val"],lo,hi,mask)}},
            "test":{"persistence":{"original_units":metrics(last_t,Y["test"],mask),"normalized":norm_metrics(last_t,Y["test"],lo,hi,mask)},
                    "tree_context":{"original_units":metrics(pt,Y["test"],mask),"normalized":norm_metrics(pt,Y["test"],lo,hi,mask)}}}
    out="res_TREE_CONTEXT"; os.makedirs(out,exist_ok=True)
    for name,p in (("pred_val",pv),("true_val",Y["val"]),("last_observed_val",last_v),
                   ("pred_test",pt),("true_test",Y["test"]),("last_observed_test",last_t)):
        np.save(f"{out}/{name}.npy",p)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
