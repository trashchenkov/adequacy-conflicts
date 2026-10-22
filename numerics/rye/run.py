"""Fixed preregistered Rye grid, bounded bisection with explicit unknown states."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,time,hashlib,sys
from pathlib import Path
from collections import deque
import numpy as np,pandas as pd
from model import solve,CALLS,ROOT,runs
TOL=.02
START=time.monotonic();DEADLINE=START+1200;ROWS=[];FRONTIERS=[];CACHE={}
x=pd.read_csv(ROOT.parents[1]/'data/rye/processed.csv').x.to_numpy()
TOTAL=json.loads((ROOT.parents[1]/'data/rye/audit.json').read_text())['consumption_kWh']
def unit(P,E,eta,delta):return dict(Pc=P,Pd=P,E=E,eta=eta,delta=delta)
CONFIGS={f'li{P}_{h}h':[unit(P,P*h,.9,0)] for P in (20,40) for h in (2,4)}
CONFIGS.update({f'tank{d}':[unit(20,60,1,d)] for d in (.005,.01,.02)})
CONFIGS.update(park=[unit(20,80,.9,0),unit(20,60,1,.01)],control_tank=[unit(20,60,1,0)],control_park=[unit(20,80,1,0),unit(20,60,1,0)])

def checkpoint():
    out={'protocol':'PROTOCOL_RU.md','elapsed_s':time.monotonic()-START,'budget_s':1200,'T':len(x),'tol_kW':TOL,'configs':CONFIGS,'frontiers':FRONTIERS,'rows':ROWS,'calls':len(CALLS),'status_counts':{str(s):sum(c['status']==s for c in CALLS) for s in sorted(set(c['status'] for c in CALLS))}}
    (ROOT/'results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')

def call(U,r,limits,label,objective='feasible',limit=8):
    if time.monotonic()>=DEADLINE or len(CALLS)>=1600:return {'status':-1,'accepted':False,'message':'preregistered_budget','witness':None}
    return solve(x,U,r,**limits,objective=objective,limit=min(limit,max(.01,DEADLINE-time.monotonic())),label=label)

def upper_zero(limits):
    b=np.maximum(-x,0);hi=0.
    H=limits.get('H');D=limits.get('D');E=limits.get('ens_lim')
    if H is not None:hi=max(hi,float(np.sort(b)[::-1][min(int(H),len(b)-1)]))
    if D is not None and D<len(b):hi=max(hi,float(np.min(np.lib.stride_tricks.sliding_window_view(b,D+1),axis=1).max()))
    if E is not None:
        l=0.;h=float(b.max())
        for _ in range(50):
            m=(l+h)/2
            if np.maximum(b-m,0).sum()<=E:h=m
            else:l=m
        hi=max(hi,h)
    return hi

def blocks(b):
    idx=np.flatnonzero(b>0)
    return np.split(idx,np.where(np.diff(idx)>1)[0]+1) if len(idx) else []

def service(a,D,P):
    if len(a)<=D:return 0.
    dp=np.full(len(a),np.inf);q=deque()
    for t,v in enumerate(a):
        while q and q[0]<t-D-1:q.popleft()
        pred=0. if t<=D else np.inf
        if q:pred=min(pred,dp[q[0]])
        if v<=P+1e-10:dp[t]=v+pred
        while q and dp[q[-1]]>=dp[t]:q.pop()
        q.append(t)
    return float(np.min(dp[-D-1:]))

def necessary(U,r,limits):
    B=np.maximum(-x-r,0);S=np.maximum(x+r,0);cap=sum(u['E'] for u in U);P=sum(u['Pd'] for u in U)
    H=limits.get('H');D=limits.get('D');E=limits.get('ens_lim')
    if E is not None and np.maximum(B-P,0).sum()>E+1e-8:return False
    lost=0;energy=0.
    for b in blocks(B):
        a=B[b]
        if H is not None:
            cheap=np.sort(a[a<=P+1e-9]);served=int(np.searchsorted(np.cumsum(cheap),cap+1e-7,side='right'));lost+=len(b)-served
        if D is not None:
            need=service(a,D,P)
            if need>cap+1e-7:return False
            energy+=need
    if H is not None and lost>H:return False
    available=max(u['eta'] for u in U)*np.minimum(S,sum(u['Pc'] for u in U)).sum()
    if D is not None and energy>available+1e-7:return False
    if H is not None:
        need=np.sort(B)[:max(0,len(B)-int(H))].sum()
        if need>available+1e-7:return False
    return True

def bounds(name,limits,label):
    key=json.dumps([name,limits],sort_keys=True)
    if key in CACHE:return dict(CACHE[key])
    U=CONFIGS[name];hi=upper_zero(limits);lo=0.
    for _ in range(20):
        m=(lo+hi)/2
        if necessary(U,m,limits):hi=m
        else:lo=m
    hi=upper_zero(limits)
    ans={'lo':max(0.,lo-1e-6),'hi':hi,'witness':None,'hi_basis':'zero_storage_analytic','lo_basis':'necessary_energy_and_power','unknown_calls':0,'limits':limits,'label':label}
    refine(name,ans,label)
    CACHE[key]=dict(ans)
    return ans

def accept(ans,a):
    if a.get('accepted'):
        ans['hi']=a['r'];ans['witness']=a['witness'];ans['hi_basis']='verified_witness';ans['witness_status']=a['status']
        for k in ['ENS','LOLH','D_linear','D_cyclic']:ans[k]=a[k]
        return True
    if a['status']==2:
        ans['lo']=a['r'];ans['lo_basis']='MILP_infeasible';return True
    ans['unknown_calls']+=1
    return False

def refine(name,ans,label,maxsteps=16):
    for it in range(maxsteps):
        if ans['hi']-ans['lo']<=TOL or time.monotonic()>=DEADLINE:break
        mid=(ans['lo']+ans['hi'])/2
        a=call(CONFIGS[name],mid,ans['limits'],f'{label}_{it}')
        if not accept(ans,a):
            l,h=ans['lo'],ans['hi']
            for point in [(l+mid)/2,(mid+h)/2]:
                if ans['lo']<point<ans['hi']:accept(ans,call(CONFIGS[name],point,ans['limits'],f'{label}_{it}q'))
            break
    ans['resolved']=ans['hi']-ans['lo']<=TOL

def joint_row(name,family,limits1,limits2,id,front=False):
    U=CONFIGS[name];L={**limits1,**limits2}
    if front:
        a={'lo':0.,'hi':0.,'resolved':True,'hi_basis':'frontier_minimum','limits':limits1};b={**a,'limits':limits2}
    else:a=bounds(name,limits1,id+'a');b=bounds(name,limits2,id+'b')
    j={'lo':max(a['lo'],b['lo']),'hi':upper_zero(L),'witness':None,'hi_basis':'zero_storage_analytic','lo_basis':'separate_minima','unknown_calls':0,'limits':L,'label':id+'j'}
    # Any existing separate witness meeting BOTH norms is a direct joint certificate.
    for v in [a,b]:
        if v.get('witness') and (L.get('H') is None or v['LOLH']<=L['H']) and (L.get('D') is None or v['D_linear']<=L['D']) and (L.get('ens_lim') is None or v['ENS']<=L['ens_lim']+2e-6):
            if v['hi']<=j['hi']:
                j.update(hi=v['hi'],witness=v['witness'],hi_basis='separate_witness_meets_both')
                for k in ['ENS','LOLH','D_linear','D_cyclic']:j[k]=v[k]
    start_r=max(a['hi'],b['hi'])
    if j['hi']>start_r+1e-9 or front:
        accept(j,call(U,start_r,L,id+'j_at_r0'))
    refine(name,j,id+'j')
    gamma=[max(0.,j['lo']-max(a['hi'],b['hi'])),max(0.,j['hi']-max(a['lo'],b['lo']))]
    row={'id':id,'family':family,'config':name,'units':U,'norms':L,'r1':a,'r2':b,'rjoint':j,'gamma':gamma,'gamma_over_sumPd':[g/sum(u['Pd'] for u in U) for g in gamma],'positive_certified':gamma[0]>1e-6,'resolved_to_tolerance':gamma[1]-gamma[0]<=TOL}
    if family!='HD':
        deltaE=sum(u['delta']*u['E'] for u in U);row['deltaE']=deltaE;row['gamma_over_tankPd']=[g/20 for g in gamma];row['gamma_over_deltaE']=[g/deltaE for g in gamma] if deltaE else None
    ROWS.append(row);checkpoint();print(id,'gamma',gamma,'r',[(v['lo'],v['hi']) for v in [a,b,j]],'elapsed',round(time.monotonic()-START,1),flush=True)


def main():
    if (ROOT/'calls.jsonl').exists():raise RuntimeError('Do not overwrite existing run')
    manifest={}
    for p in [ROOT/'PROTOCOL_RU.md',ROOT.parents[1]/'data/rye/processed.csv',ROOT.parents[1]/'data/rye/rye_generation_and_load.csv',ROOT/'model.py',Path(__file__),ROOT.parents[1]/'src/ref_milp.py']:manifest[str(p.relative_to(ROOT.parents[1]))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    # HD has extensive cached separate optima, no grid expansion.
    for name in list(CONFIGS)[:4]:
        for H in [72,150]:
            for D in [24,12,8]:joint_row(name,'HD',{'H':H},{'D':D},f'{name}_H{H}D{D}')
    # Fixed norms first, so a hard frontier cannot consume all remaining time.
    for name in list(CONFIGS)[4:]:
        for H in [72,150]:
            for frac in [.001,.01]:joint_row(name,'EH',{'ens_lim':TOTAL*frac},{'H':H},f'{name}_H{H}E{frac}')
    for name in list(CONFIGS)[4:]:
        U=CONFIGS[name]
        en=call(U,0,{},name+'_frontENS',objective='ENS',limit=45)
        hh=call(U,0,{},name+'_frontH',objective='H',limit=45)
        f={'config':name,'r_reference':0.,'ens':en,'H':hh,'established':en['status']==0 and hh['status']==0}
        FRONTIERS.append(f);checkpoint()
        if f['established']:
            joint_row(name,'EH_frontier',{'ens_lim':en['ENS']+1e-6},{'H':hh['LOLH']},name+'_frontier',front=True)
        else:print(name,'frontier not established',en['status'],hh['status'],flush=True)
    checkpoint()
if __name__=='__main__':main()
