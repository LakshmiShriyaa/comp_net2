"""Per-OD ridge with topology-aware source, destination, and path-load context."""
import json
import os
import numpy as np
import networkx as nx
from data import load_tm, despike, prepare, metrics, norm_metrics


def route_incidence(n, edge_file):
    graph = nx.Graph(); graph.add_nodes_from(range(n))
    with open(edge_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                u,v=map(int,line.split()[:2]); graph.add_edge(u,v)
    edges=[tuple(sorted(e)) for e in graph.edges()]
    eidx={e:k for k,e in enumerate(edges)}
    paths=[]
    A=np.zeros((n*n,len(edges)),dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i==j: paths.append([]); continue
            path=nx.shortest_path(graph,i,j)
            ids=[eidx[tuple(sorted((u,v)))] for u,v in zip(path[:-1],path[1:])]
            paths.append(ids)
            A[i*n+j,ids]=1
    return A,paths


def context_features(X, A, paths, row_scale, col_scale, path_scale, n):
    s,w=X.shape[:2]
    row=np.log1p(X.sum(axis=3)/row_scale[None,None,:])
    col=np.log1p(X.sum(axis=2)/col_scale[None,None,:])
    flat=X.reshape(s*w,n*n)
    link=flat@A
    avg=np.zeros((s*w,n*n),dtype=np.float32)
    mx=np.zeros_like(avg)
    for q,p in enumerate(paths):
        if p:
            avg[:,q]=link[:,p].mean(axis=1)
            mx[:,q]=link[:,p].max(axis=1)
    avg=np.log1p(avg.reshape(s,w,n,n)/path_scale[0][None,None])
    mx=np.log1p(mx.reshape(s,w,n,n)/path_scale[1][None,None])
    row=np.broadcast_to(row[:,:,:,None],(s,w,n,n))
    col=np.broadcast_to(col[:,:,None,:],(s,w,n,n))
    return np.stack([row,col,avg,mx],axis=2)


def fit_predict(Xtr,Ytr,Xev,scale,ctxtr,ctxev,k,alpha,n):
    pred=np.zeros((len(Xev),n,n),dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            s=scale[i,j]
            owntr=Xtr[:,-k:,i,j]/s; ownev=Xev[:,-k:,i,j]/s
            x=np.concatenate([owntr,ctxtr[:,-3:,:,i,j].reshape(len(Xtr),-1)],axis=1)
            xe=np.concatenate([ownev,ctxev[:,-3:,:,i,j].reshape(len(Xev),-1)],axis=1)
            y=(Ytr[:,i,j]-Xtr[:,-1,i,j])/s
            xm,ym=x.mean(0),y.mean(); xc,yc=x-xm,y-ym
            beta=np.linalg.solve(xc.T@xc+alpha*np.eye(x.shape[1]),xc.T@yc)
            b=ym-xm@beta
            pred[:,i,j]=np.maximum(Xev[:,-1,i,j]+(xe@beta+b)*s,0)
    return pred


def main():
    n,w,h=23,12,4
    tm,_=despike(load_tm("geant_final.npy",n))
    times=np.load("geant_times.npy")
    d=prepare(tm,w,h,mask_diag=True,times=times,step_min=15,season=96)
    sc=d["scaler"]
    X={k:sc.inverse(d[k][0]) for k in ("train","val","test")}
    Y={k:sc.inverse(d[k][1][:,-1]) for k in ("train","val","test")}
    tr=d["tm"][:d["t_end"]]
    scale=np.maximum(tr.max(0)-tr.min(0),1.0)
    row_scale=np.maximum(tr.sum(axis=2).mean(0),1.0)
    col_scale=np.maximum(tr.sum(axis=1).mean(0),1.0)
    A,paths=route_incidence(n,"geant_edges.txt")
    # Training-only normalizers for path context features.
    l=tr.reshape(len(tr),n*n)@A
    pavg=np.ones(n*n); pmax=np.ones(n*n)
    for q,path in enumerate(paths):
        if path:
            pavg[q]=max(float(l[:,path].mean()),1.0)
            pmax[q]=max(float(l[:,path].max(axis=1).mean()),1.0)
    path_scale=(pavg.reshape(n,n),pmax.reshape(n,n))
    ctx={k:context_features(X[k],A,paths,row_scale,col_scale,path_scale,n) for k in X}
    mask=~np.eye(n,dtype=bool); lo,hi=tr.min(0),tr.max(0)
    best=np.full((n,n),np.inf); best_k=np.zeros((n,n),int); best_a=np.zeros((n,n))
    pred_v=np.zeros_like(Y["val"],dtype=np.float64)
    for k in (3,6,12):
        for alpha in (.1,1,10,100):
            pv=fit_predict(X["train"],Y["train"],X["val"],scale,ctx["train"],ctx["val"],k,alpha,n)
            loss=(((pv-Y["val"])/np.maximum(hi-lo,1))**2).mean(0)
            use=loss<best
            best[use],best_k[use],best_a[use],pred_v[:,use]=loss[use],k,alpha,pv[:,use]
        print(f"finished validation lag count {k}")
    Xfit=np.concatenate([X["train"],X["val"]]); Yfit=np.concatenate([Y["train"],Y["val"]])
    ctxfit=np.concatenate([ctx["train"],ctx["val"]])
    pred_t=np.zeros_like(Y["test"],dtype=np.float64)
    for k in (3,6,12):
        for alpha in (.1,1,10,100):
            use=(best_k==k)&(best_a==alpha)&mask
            if not use.any(): continue
            # Fit the complete grid once, then retain only selected OD predictions.
            p=fit_predict(Xfit,Yfit,X["test"],scale,ctxfit,ctx["test"],k,alpha,n)
            pred_t[:,use]=p[:,use]
    last_v,last_t=X["val"][:,-1].copy(),X["test"][:,-1].copy()
    for p in (pred_v,pred_t,last_v,last_t): p[:,np.arange(n),np.arange(n)]=0
    report={"selection":"per-OD history length and ridge penalty selected on validation normalized RMSE",
            "context":"source outgoing total, destination incoming total, average and peak shortest-path link load over last 3 observations; all scales from training only",
            "validation":{"persistence":{"original_units":metrics(last_v,Y["val"],mask),"normalized":norm_metrics(last_v,Y["val"],lo,hi,mask)},
                         "context_ridge":{"original_units":metrics(pred_v,Y["val"],mask),"normalized":norm_metrics(pred_v,Y["val"],lo,hi,mask)}},
            "test":{"persistence":{"original_units":metrics(last_t,Y["test"],mask),"normalized":norm_metrics(last_t,Y["test"],lo,hi,mask)},
                    "context_ridge":{"original_units":metrics(pred_t,Y["test"],mask),"normalized":norm_metrics(pred_t,Y["test"],lo,hi,mask)}}}
    out="res_FLOW_CONTEXT"; os.makedirs(out,exist_ok=True)
    for name,p in (("pred_val",pred_v),("true_val",Y["val"]),("last_observed_val",last_v),
                   ("pred_test",pred_t),("true_test",Y["test"]),("last_observed_test",last_t)):
        np.save(f"{out}/{name}.npy",p)
    np.savez_compressed(f"{out}/params.npz",lag=best_k,alpha=best_a)
    with open(f"{out}/metrics.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
