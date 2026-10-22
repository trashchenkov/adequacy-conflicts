"""Rye normalized storage LP/MILP; no dependency on previous solver builders."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
ROOT=Path(__file__).resolve().parent
CALLS=[]

def runs(a,cyclic=False):
    a=np.asarray(a,bool)
    if np.all(a):return len(a)
    best=cur=0
    for v in np.r_[a,a] if cyclic else a:
        cur=cur+1 if v else 0;best=max(best,cur)
    return best

def solve(x,units,r,H=None,D=None,ens_lim=None,objective='feasible',limit=8.,label='test',save=True):
    st=time.monotonic();x=np.asarray(x,float);T=len(x);n=len(units);q=n*T
    c=np.arange(q).reshape(n,T);d=c+q;e=c+2*q;ell=np.arange(3*q,3*q+T)
    S=np.maximum(x+r,0);B=np.maximum(-x-r,0);ts=np.flatnonzero(B>0)
    binary=H is not None or D is not None or objective=='H'
    z=np.full(T,-1,int)
    if binary:z[ts]=np.arange(3*q+T,3*q+T+len(ts))
    nv=3*q+T+(len(ts) if binary else 0)
    lower=np.zeros(nv);upper=np.full(nv,np.inf);integral=np.zeros(nv)
    col=lambda k:np.array([u[k] for u in units])[:,None]
    upper[c.ravel()]=np.broadcast_to(col('Pc'),(n,T)).ravel()*(np.tile(S>0,n))
    upper[d.ravel()]=np.broadcast_to(col('Pd'),(n,T)).ravel()*(np.tile(B>0,n))
    upper[e.ravel()]=np.broadcast_to(col('E'),(n,T)).ravel();upper[ell]=B
    if binary:upper[z[ts]]=1;integral[z[ts]]=1
    rr=[];cc=[];vv=[];lo=[];hi=[]
    def add(cols,vals,l,h):
        k=len(lo);rr.extend([k]*len(cols));cc.extend(cols);vv.extend(vals);lo.append(l);hi.append(h)
    for i,u in enumerate(units):
        for t in range(T):add([e[i,t],e[i,(t-1)%T],c[i,t],d[i,t]],[1,-(1-u['delta']),-u['eta'],1],0,0)
    for t in range(T):
        add(list(d[:,t])+[ell[t]],[1]*(n+1),B[t],B[t])
        if S[t]>0:add(list(c[:,t]),[1]*n,-np.inf,S[t])
    if binary:
        for t in ts:add([ell[t],z[t]],[1,-B[t]],-np.inf,0)
        if H is not None:add(list(z[ts]),[1]*len(ts),-np.inf,H)
        if D is not None and D<T:
            for t in range(T-D):
                ids=z[t:t+D+1]
                if np.all(ids>=0):add(list(ids),[1]*(D+1),-np.inf,D)
    if ens_lim is not None:add(list(ell),[1]*T,-np.inf,ens_lim)
    A=coo_matrix((vv,(rr,cc)),shape=(len(lo),nv)).tocsc();cost=np.zeros(nv)
    if objective=='ENS':cost[ell]=1
    elif objective=='H':cost[z[ts]]=1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        res=milp(cost,integrality=integral,bounds=Bounds(lower,upper),constraints=LinearConstraint(A,lo,hi),options={'time_limit':max(.01,limit),'threads':1,'mip_rel_gap':0.,'mip_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    out={'label':label,'r':float(r),'H':H,'D':D,'ens_lim':ens_lim,'objective':objective,'status':int(res.status),'message':res.message,'elapsed_s':time.monotonic()-st,'accepted':False,'objective_value':None if res.fun is None else float(res.fun),'dual_bound':None if getattr(res,'mip_dual_bound',None) is None else float(res.mip_dual_bound),'witness':None}
    if res.x is not None:
        v=res.x;cv=v[c];dv=v[d];ev=v[e];lv=v[ell]
        mask=np.zeros(T,bool)
        if binary:mask[ts]=np.rint(v[z[ts]]).astype(bool)
        else:mask=lv>1e-7
        dyn=ev-(1-col('delta'))*np.roll(ev,1,axis=1)-col('eta')*cv+dv
        phys=max(float(np.max(np.abs(dyn))),float(np.max(np.abs(dv.sum(0)+lv-B))),float(np.max(cv.sum(0)-S)),float(np.max(lower-v)),float(np.max(v-upper)),float(np.max(lv[~mask],initial=0)) if binary else 0.)
        integ=float(np.max(np.abs(v[z[ts]]-np.rint(v[z[ts]])),initial=0)) if binary else 0.
        hh=int(mask.sum());dd=runs(mask);ens=float(lv.sum())
        valid=phys<=2e-6 and integ<=1e-6 and (H is None or hh<=H) and (D is None or dd<=D) and (ens_lim is None or ens<=ens_lim+2e-6)
        out.update(accepted=bool(valid),physical_residual=phys,integrality_residual=integ,ENS=ens,LOLH=hh,D_linear=dd,D_cyclic=runs(mask,True))
        if valid and save:
            (ROOT/'witnesses').mkdir(exist_ok=True)
            name=f'{len(CALLS):04d}_{label}.npz';path=ROOT/'witnesses'/name
            np.savez_compressed(path,x=x,c=cv,d=dv,e=ev,ell=lv,r=r,allowed_loss=mask,units_json=json.dumps(units),limits_json=json.dumps({'H':H,'D':D,'ens_lim':ens_lim}),metadata_json=json.dumps(out))
            out['witness']=str(path.relative_to(ROOT))
    CALLS.append(out)
    if save:
        with (ROOT/'calls.jsonl').open('a') as f:f.write(json.dumps(out,ensure_ascii=False)+'\n')
    return out
