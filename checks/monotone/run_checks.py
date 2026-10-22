
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Reproducible independent numerical checks of the monotone note (delta=0)."""
from pathlib import Path
import argparse, json, time, sys
from fractions import Fraction as F
import numpy as np
import scipy
from independent_model import Case, dispatch, capacity, bound_B, CALLS, runs

ROOT=Path(__file__).resolve().parent


def save(name,data,out):
    (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def random_case(rng,equal):
    T=int(rng.integers(4,11)); n=int(rng.integers(1 if equal else 2,4))
    x=(rng.integers(1,17,T)*.25*rng.choice([-1,1],T,p=[.65,.35]))
    x[0]=abs(x[0]); x[-1]=-abs(x[-1])
    pc=rng.integers(1,13,n)*.25; pd=rng.integers(1,13,n)*.25
    E=rng.integers(1,25,n)*.25
    choices=np.array([.35,.5,.65,.8,.9,1.])
    eta=np.full(n,rng.choice(choices)) if equal else rng.choice(choices,n)
    if not equal and np.ptp(eta)==0:
        eta[-1]=.35 if eta[0]>.35 else 1.
    return Case(x.tolist(),E.tolist(),pc.tolist(),pd.tolist(),eta.tolist())


def dominance(count,equal):
    rng=np.random.default_rng(2026102101 if equal else 2026102102)
    rows=[]
    for k in range(count):
        case=random_case(rng,equal); T=len(case.x); n=len(case.E)
        r=float(rng.uniform(0,.25)*max(-min(case.x),0)) if k%3 else 0.
        f=np.full(T,r) if k%2==0 else rng.uniform(0,r,T)
        obj=rng.uniform(-1,1,3*n*T+T)
        sample=dispatch(case,r,output=f,objective=obj,label=f'dom:{equal}:{k}:sample')
        row={'case':case.record(),'r':r,'objective':obj.tolist(),'sample':sample}
        if sample['status']==0:
            base=dispatch(case,r,label=f'dom:{equal}:{k}:base')
            restricted=dispatch(case,r,shortage_cap=sample['ell'],
                                label=f'dom:{equal}:{k}:restricted')
            row.update(base=base,restricted=restricted)
            if base['status']==restricted['status']==0:
                tol=1e-7*max(1.,sum(max(-a-r,0) for a in case.x))
                diff=restricted['ens']-base['ens']
                row.update(difference=diff,tolerance=tol,mismatch=diff>tol,
                           physical_ok=max(z['violation'] for z in [sample,base,restricted])<=tol)
            else: row['unknown']=True
        else: row['unknown']=True
        rows.append(row)
        if (k+1)%50==0: print('dominance',equal,k+1,flush=True)
    return rows


def bound_tests(count):
    rng=np.random.default_rng(2026102103);rows=[]
    for k in range(count):
        case=random_case(rng,False)
        base=dispatch(case,0.,label=f'bound:{k}:base')
        if base['status']!=0:
            rows.append({'case':case.record(),'unknown':True,'failed':base});continue
        Ebar=float(base['ens']*rng.uniform(.3,1.05))
        D=int(rng.integers(1,len(case.x)-1)); cyclic=bool(k%2)
        e=capacity(case,ens_cap=Ebar,label=f'bound:{k}:E')
        d=capacity(case,run_cap=D,cyclic=cyclic,label=f'bound:{k}:D')
        j=capacity(case,ens_cap=Ebar,run_cap=D,cyclic=cyclic,label=f'bound:{k}:joint')
        B=bound_B(case)
        row={'case':case.record(),'Ebar':Ebar,'D':D,'cyclic':cyclic,
             'rE':e,'rD':d,'joint':j,'bound':B}
        if all(a['status']!='unknown' for a in [e,d,j]) and B['status']==0:
            lo=max(0.,j['lo']-max(e['hi'],d['hi']))
            hi=max(0.,j['hi']-max(e['lo'],d['lo']))
            row.update(gamma_interval=[lo,hi],bound_violated=lo>B['B']+2e-6,
                       bound_verified=hi<=B['B']+2e-6)
        else: row['unknown']=True
        rows.append(row)
        print('bound',k+1,row.get('gamma_interval'),B.get('B'),flush=True)
    return rows


def examples():
    peak=Case([1,-1,-1],[1],[1],[1],[1])
    h=dispatch(peak,0.,hours_cap=1,label='example:peak:H')
    p=dispatch(peak,0.,shortage_cap=[0,.5,.5],label='example:peak:P')
    hp=dispatch(peak,0.,hours_cap=1,shortage_cap=[0,.5,.5],label='example:peak:joint')
    rpeak=capacity(peak,hours_cap=1,shortage_cap=[0,.5,.5],label='example:peak:capacity')
    assert h['status']==p['status']==0 and hp['status']==2
    # Rational joint lower bound: 2-2r-.5 <= 1 gives r>=1/4.
    rational={'peak_joint_r':str(F(1,4)),'peak_joint_discharge':[str(F(3,4)),str(F(1,4))],
              'peak_joint_ell':['0','0',str(F(1,2))]}
    assert (1-F(1,4))+(1-F(1,4)-F(1,2))==1
    lol=Case([3,-2,-1,-2],[3],[3],[1],[1])
    a=dispatch(lol,0.,label='example:LOLF:ENS')
    assert a['status']==0 and abs(a['ens']-2)<1e-9
    # Enumerate actual connected supports. Positive-support lower bound epsilon
    # is explicit: upper-bound indicators alone cannot encode exact event count.
    eps_checks=[]
    for eps in [1e-2,1e-4,1e-6]:
        candidates=[]
        for start in range(1,4):
            for end in range(start,4):
                cap=np.zeros(4);cap[start:end+1]=[2,1,2][start-1:end]
                floor=np.zeros(4);floor[start:end+1]=eps
                z=dispatch(lol,0.,shortage_cap=cap,shortage_floor=floor,
                           label=f'example:LOLF:epsilon:{eps}:{start}:{end}')
                candidates.append(z)
        feasible=[z for z in candidates if z['status']==0]
        eps_checks.append({'epsilon':eps,'best_ens':min(z['ens'] for z in feasible),
                           'solutions':candidates})
    rational_family=[]
    for r in [F(1,10),F(1,100),F(1,10000)]:
        c=3-2*r; ds=[F(1),1-2*r,F(1)]
        ls=[0,1-r,r,1-r]
        assert sum(ds)==c and sum(ls)==2-r
        assert 0<c<=3 and all(0<=v<=1 for v in ds)
        assert all(v>0 for v in ls[1:])
        rational_family.append({'r':str(r),'c':str(c),'d':list(map(str,ds)),
                                'ell':list(map(str,ls)),'ENS':str(sum(ls))})
    # Known unequal-efficiency positive control, independent from random sample.
    known=Case([1,1,-1.5,-3],[2,2],[1,1],[1,1],[.8,.95])
    kbase=dispatch(known,0,label='control:heter:base')
    kh=dispatch(known,0,hours_cap=1,label='control:heter:H')
    kres=dispatch(known,0,shortage_cap=kh['ell'],label='control:heter:restricted')
    assert kres['ens']-kbase['ens']>1e-3
    return {'peak_vs_hours':{'case':peak.record(),'H':h,'peak':p,'joint':hp,
                             'capacity':rpeak,'rational':rational},
            'ENS_vs_events':{'case':lol.record(),'ens':a,'epsilon_checks':eps_checks,
                             'exact_family':rational_family},
            'heterogeneous_positive_control':{'case':known.record(),'base':kbase,
                                              'hours':kh,'restricted':kres}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();out=ROOT/('smoke' if args.smoke else 'results')
    out.mkdir(exist_ok=True);start=time.time()
    ex=examples();save('examples.json',ex,out)
    eq=dominance(5 if args.smoke else 500,True);save('equal.json',eq,out)
    ne=dominance(4 if args.smoke else 200,False);save('unequal.json',ne,out)
    bt=bound_tests(2 if args.smoke else 30);save('bound.json',bt,out)
    save('solver_calls.json',CALLS,out)
    summary={'python':sys.version,'scipy':scipy.__version__,'numpy':np.__version__,
             'runtime_seconds':time.time()-start,'equal':{
               'n':len(eq),'mismatches':sum(a.get('mismatch',False) for a in eq),
               'unknown':sum(a.get('unknown',False) for a in eq),
               'max_abs_difference':max(abs(a.get('difference',0)) for a in eq)},
             'unequal':{'n':len(ne),'mismatches':sum(a.get('mismatch',False) for a in ne),
               'unknown':sum(a.get('unknown',False) for a in ne),
               'max_difference':max(a.get('difference',0) for a in ne)},
             'bound':{'n':len(bt),'violations':sum(a.get('bound_violated',False) for a in bt),
               'verified':sum(a.get('bound_verified',False) for a in bt),
               'unknown':sum(a.get('unknown',False) for a in bt)},
             'calls':len(CALLS),'status_counts':{str(k):sum(a['status']==k for a in CALLS)
                          for k in sorted(set(a['status'] for a in CALLS))},
             'max_physical_violation':max(a.get('violation',0) for a in CALLS),
             'max_integer_violation':max(a.get('integer_violation',0) for a in CALLS)}
    save('summary.json',summary,out)
    save('unknown.json',[a for a in CALLS if a['status'] not in (0,2)],out)
    save('nonoptimal_statuses.json',[a for a in CALLS if a['status']!=0],out)
    save('dominance_mismatches.json',{'equal':[a for a in eq if a.get('mismatch')],
                                    'unequal':[a for a in ne if a.get('mismatch')]},out)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
