"""Bounded fixed-source H/D study. Independent sparse physical formulation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import csv, json, time, hashlib, warnings, argparse
from pathlib import Path
from collections import deque
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]/"data/yakutia"
CALLS=[]

def runs(mask,cyclic=True):
    a=np.asarray(mask,bool)
    if a.all(): return len(a)
    best=cur=0
    for v in np.r_[a,a] if cyclic else a:
        cur=cur+1 if v else 0;best=max(best,cur)
    return best

def segments(x,r):
    d=np.maximum(-x-r,0); good=np.where(d==0)[0]
    if not len(good): return d,[np.arange(len(x))],True
    order=(np.arange(len(x))+good[0])%len(x)
    out=[]; cur=[]
    for t in order:
        if d[t]>0:cur.append(t)
        elif cur:out.append(np.array(cur));cur=[]
    if cur:out.append(np.array(cur))
    return d,out,False

def min_service_energy(a,D,P):
    """Relaxation: minimum energy to interrupt every run of D+1 losses."""
    n=len(a)
    if n<=D:return 0.
    costs=np.full(n,np.inf);q=deque()
    for t,v in enumerate(a):
        while q and q[0]<t-D-1:q.popleft()
        pred=0. if t<=D else (costs[q[0]] if q else np.inf)
        if q:pred=min(pred,costs[q[0]])
        if v<=P+1e-10:costs[t]=pred+v
        while q and costs[q[-1]]>=costs[t]:q.pop()
        q.append(t)
    return float(np.min(costs[max(0,n-D-1):]))

def necessary(x,e,r,H=None,D=None):
    deficit,blocks,allbad=segments(x,r)
    cap=sum(e['E']);power=sum(e['Pd']);surplus=np.maximum(x+r,0)
    if allbad:return False if H is None or H<len(x) else True
    if H is not None:
        lost=0
        for b in blocks:
            a=np.sort(deficit[b]);a=a[a<=power+1e-9]
            served=np.searchsorted(np.cumsum(a),cap+1e-7,side='right')
            lost+=len(b)-served
        if lost>H:return False
        a=np.sort(deficit)
        if sum(a[:max(0,len(a)-H)])>max(e['eta'])*sum(surplus)+1e-7:return False
    if D is not None:
        total=0.
        for b in blocks:
            need=min_service_energy(deficit[b],D,power)
            if need>cap+1e-7:return False
            total+=need
        if total>max(e['eta'])*sum(surplus)+1e-7:return False
    return True

def analytic_bounds(x,e,H=None,D=None):
    lo=0.;hi=float(max(0,-min(x)))
    for _ in range(18):
        m=(lo+hi)/2
        if necessary(x,e,m,H,D):hi=m
        else:lo=m
    a=np.maximum(-x,0)
    upper=0.
    if H is not None:upper=max(upper,float(np.sort(a)[::-1][min(H,len(a)-1)]))
    if D is not None and D<len(a):
        upper=max(upper,float(np.max(np.min(np.lib.stride_tricks.sliding_window_view(np.r_[a,a[:D]],D+1),axis=1))))
    return {'lo':max(0.,lo-1e-5),'hi':upper,'lo_source':'necessary_block_energy_and_global_energy','hi_source':'zero_storage_witness','calls':[]}

def solve(x,e,r,H=None,D=None,limit=2.,fixed_zero=False,label=''):
    start=time.monotonic();n=len(e['E']);T=len(x);q=n*T
    c=np.arange(q).reshape(n,T);d=c+q;s=c+2*q;ell=np.arange(3*q,3*q+T)
    S=np.maximum(x+r,0);B=np.maximum(-x-r,0);times=np.flatnonzero(B>1e-10)
    discrete=not fixed_zero
    z=np.full(T,-1,dtype=int)
    if discrete:z[times]=np.arange(3*q+T,3*q+T+len(times))
    nv=3*q+T+(len(times) if discrete else 0)
    lb=np.zeros(nv);ub=np.full(nv,np.inf);integ=np.zeros(nv)
    ub[c.ravel()]=(np.array(e['Pc'])[:,None]*(S>0)).ravel()
    ub[d.ravel()]=(np.array(e['Pd'])[:,None]*(B>0)).ravel()
    ub[s.ravel()]=np.broadcast_to(np.array(e['E'])[:,None],(n,T)).ravel()
    ub[ell]=0 if fixed_zero else B
    if discrete:ub[z[times]]=1;integ[z[times]]=1
    rr=[];cc=[];vv=[];lower=[];upper=[]
    def add(cols,vals,l,u):
        ix=len(lower);rr.extend([ix]*len(cols));cc.extend(cols);vv.extend(vals);lower.append(l);upper.append(u)
    for i in range(n):
        for t in range(T):add([s[i,t],s[i,(t-1)%T],c[i,t],d[i,t]],[1,-1,-e['eta'][i],1],0,0)
    for t in range(T):
        add(list(d[:,t])+[ell[t]],[1]*(n+1),B[t],B[t])
        if S[t]>0:add(list(c[:,t]),[1]*n,-np.inf,S[t])
    if discrete:
        for t in times:add([ell[t],z[t]],[1,-B[t]],-np.inf,0)
        if H is not None:add(list(z[times]),[1]*len(times),-np.inf,H)
        if D is not None and D<T:
            # A zero-deficit hour breaks the series already.
            for t in times:
                ids=z[(t+np.arange(D+1))%T]
                if np.all(ids>=0):add(list(ids),[1]*len(ids),-np.inf,D)
    A=coo_matrix((vv,(rr,cc)),shape=(len(lower),nv)).tocsc()
    obj=np.zeros(nv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        res=milp(obj,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(A,lower,upper),options={'time_limit':limit,'mip_rel_gap':0.,'threads':1,'mip_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    out={'label':label,'r':r,'H':H,'D':D,'status':int(res.status),'message':res.message,'elapsed_s':time.monotonic()-start,'accepted':False}
    witness=None
    if res.x is not None:
        a=res.x;cv=a[c];dv=a[d];ev=a[s];lv=a[ell]
        # Direct physical residuals, independent of constraint matrix assembly.
        viol=max(np.max(np.abs(ev-np.roll(ev,1,axis=1)-np.array(e['eta'])[:,None]*cv+dv)),np.max(np.abs(dv.sum(axis=0)+lv-B)),np.max(cv.sum(axis=0)-S),np.max(np.maximum(-a,0)),np.max(np.maximum(a-ub,0)))
        mask=(np.rint(a[z[times]]).astype(int)==1) if discrete else np.zeros(len(times),bool)
        allowed=np.zeros(T,bool);allowed[times]=mask
        h=int(sum(allowed));dc=runs(allowed);dl=runs(allowed,False)
        iv=float(np.max(np.abs(a[z[times]]-np.rint(a[z[times]])))) if discrete and len(times) else 0.
        valid=viol<2e-5 and iv<1e-6 and (H is None or h<=H) and (D is None or dc<=D)
        out.update(violation=float(viol),integer_violation=iv,hours=h,longest_cyclic=dc,longest_linear=dl,accepted=bool(valid))
        if valid:witness={'c':cv,'d':dv,'soc':ev,'ell':lv,'allowed_loss':allowed,'r':np.array(r)}
    CALLS.append(out)
    return out,witness

def update(b,o,w,path):
    b['calls'].append(len(CALLS)-1)
    if o['accepted'] and o['r']<b['hi']:
        b['hi']=o['r'];b['hi_source']='checked_MILP_witness';b['witness']=str(path.relative_to(HERE));b['hours']=o['hours'];b['longest_cyclic']=o['longest_cyclic'];b['longest_linear']=o['longest_linear']
        np.savez_compressed(path,**w)
    elif o['status']==2:b['lo']=max(b['lo'],o['r']-1e-5);b['lo_source']='MILP_infeasible_or_necessary_bounds'

def refine(x,e,b,H,D,seconds,label,root):
    end=time.monotonic()+seconds
    while b['hi']-b['lo']>.1 and time.monotonic()<end:
        m=(b['lo']+b['hi'])/2
        o,w=solve(x,e,m,H,D,limit=max(.2,min(2.,end-time.monotonic())),fixed_zero=H==0,label=label)
        update(b,o,w,root/(label+'.npz'))
        if not o['accepted'] and o['status']!=2:break
    b['status']='resolved_0.1kW' if b['hi']-b['lo']<=.1 else 'interval_unresolved'

def main():
    start=time.monotonic();deadline=start+720
    equipment=json.loads((BASE/'online_pilot_2026-10-16/equipment.json').read_text())['cases']
    datasets=[]
    with np.load(BASE/'online_pilot_2026-10-16/data/winter_1008.npz') as a:
        datasets.append(('winter2023',a['x_model_kW'][-1],1008))
    with np.load(BASE/'online_pilot_2026-10-16/data/annual_8760.npz') as a:
        years=a['years'];xx=a['x_model_kW']
        for i in [18]+list(range(18)):datasets.append((f'annual{years[i]}',xx[i],8760))
    profiles=json.loads((BASE/'five_storage_parameter_grid/cases.json').read_text())['profiles']
    for name,p in profiles.items():
        load=np.array(p['load']);gen=np.array(p['generation_R1'])
        # Previous parameter profile normalization R=0.4, retained explicitly.
        datasets.append(('pilot72_'+name,.4*gen-load,len(load)))
    root=HERE/'witnesses';root.mkdir(exist_ok=True)
    output=[]
    for name,xbase,T in datasets:
        for eq in equipment:
            x=xbase-eq['aux_load_kw'];e=eq['aggregated'];rows=[]
            for annualH,D in [(72,24),(35,8),(35,12)]:
                H=int(np.floor(annualH*T/8760))
                rows.append({'id':f'{name}_{eq["id"]}_H{H}D{D}','profile':name,'equipment':eq['id'],'T':T,'H_annual':annualH,'H':H,'D':D,'r_H':analytic_bounds(x,e,H=H),'r_D':analytic_bounds(x,e,D=D),'r_joint':analytic_bounds(x,e,H=H,D=D),'sum_Pd':sum(e['Pd']),'structural_zero':H<=D})
            # Each data/equipment receives a bounded share, not a new search grid.
            end=min(deadline,time.monotonic()+(18 if T==8760 else 10))
            for row in rows:
                for key,h,d in [('r_H',row['H'],None),('r_D',None,row['D'])]:
                    left=end-time.monotonic()
                    if left>0:refine(x,e,row[key],h,d,min(2.2,left),row['id']+'_'+key,root)
                if row['structural_zero']:
                    row['r_joint']=dict(row['r_H']);row['gamma']=[0.,0.];row['gamma_basis']='H<=D implies joint feasible set equals H feasible set'
                else:
                    b=row['r_joint'];b['lo']=max(b['lo'],row['r_H']['lo'],row['r_D']['lo'])
                    h=row['r_H']
                    if h.get('longest_cyclic',10**9)<=row['D']:
                        b['hi']=h['hi'];b['hi_source']='H_witness_also_satisfies_D';b['witness']=h['witness']
                    left=end-time.monotonic()
                    if left>0 and b['hi']-b['lo']>.1:refine(x,e,b,row['H'],row['D'],min(2.2,left),row['id']+'_r_joint',root)
                    row['gamma']=[max(0.,b['lo']-max(row['r_H']['hi'],row['r_D']['hi'])),max(0.,b['hi']-max(row['r_H']['lo'],row['r_D']['lo']))]
                    row['gamma_basis']='independent capacity intervals'
                row['gamma_fraction_Pd']=[v/row['sum_Pd'] for v in row['gamma']]
                output.append(row)
            (HERE/'results.json').write_text(json.dumps({'rows':output,'elapsed_s':time.monotonic()-start,'calls':CALLS},ensure_ascii=False,indent=2))
            print(name,eq['id'],'elapsed',round(time.monotonic()-start,1),'gamma',[r['gamma'] for r in rows],flush=True)
    files=[BASE/'online_pilot_2026-10-16/data/annual_8760.npz',BASE/'online_pilot_2026-10-16/data/winter_1008.npz',BASE/'online_pilot_2026-10-16/equipment.json',BASE/'five_storage_parameter_grid/cases.json',BASE/'passport_search_2026-10-14/catalogue_comparison.json',BASE/'catalog_significance_2026-10-14/summary.json',HERE/'PROTOCOL_RU.md',Path(__file__)]
    # Data live under data/yakutia; the script does not. Keys stay relative to whichever root contains the file.
    keys={}
    for p in files:
        try:
            key=str(p.resolve().relative_to(BASE.resolve()))
        except ValueError:
            key=str(p.resolve().relative_to(HERE.parents[1].resolve()))
        keys[key]=hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE/'manifest.json').write_text(json.dumps(keys,indent=2)+'\n')
    with (HERE/'table.csv').open('w') as f:
        fields=['id','T','H','D','rH_lo','rH_hi','rD_lo','rD_hi','rJ_lo','rJ_hi','gamma_lo','gamma_hi','structural_zero']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for a in output:w.writerow(dict(id=a['id'],T=a['T'],H=a['H'],D=a['D'],rH_lo=a['r_H']['lo'],rH_hi=a['r_H']['hi'],rD_lo=a['r_D']['lo'],rD_hi=a['r_D']['hi'],rJ_lo=a['r_joint']['lo'],rJ_hi=a['r_joint']['hi'],gamma_lo=a['gamma'][0],gamma_hi=a['gamma'][1],structural_zero=a['structural_zero']))

if __name__=='__main__':main()
