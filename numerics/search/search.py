"""All sign structures, exact support enumeration, heuristic continuous search."""
import argparse, ctypes, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parent
# Build the native accelerator from source; never ship a platform binary.
import subprocess, sys
CODE=Path(__file__).resolve().parent
library=CODE/('engine.dylib' if sys.platform=='darwin' else 'engine.so')
if not library.exists() or library.stat().st_mtime < (CODE/'engine.cpp').stat().st_mtime:
    subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(CODE/'engine.cpp'),'-o',str(library)],check=True)
LIB=ctypes.CDLL(str(library))
DP=ctypes.POINTER(ctypes.c_double)
LIB.evaluate.argtypes=[DP,ctypes.c_int,ctypes.c_double,ctypes.c_double,DP,DP]

def evaluate(x,E,Pc,table=False):
    a=np.ascontiguousarray(x,dtype=np.float64); out=np.zeros(14)
    tab=np.full(1<<len(x),np.nan) if table else None
    LIB.evaluate(a.ctypes.data_as(DP),len(a),E,Pc,out.ctypes.data_as(DP),tab.ctypes.data_as(DP) if table else None)
    ans=[]
    for cyclic,z in enumerate(out.reshape(2,7)):
        ans.append(dict(cyclic=bool(cyclic),gamma=float(z[0]),H=int(z[1]),D=int(z[2]),r_H=float(z[3]),r_D=float(z[4]),r_joint=float(z[5])))
    return (ans,tab) if table else ans

def runs(mask,T,cyclic):
    if mask==(1<<T)-1:return T
    best=cur=0
    for t in range(T*(2 if cyclic else 1)):
        cur=cur+1 if mask&(1<<(t%T)) else 0;best=max(cur,best)
    return best

def pairs(neg,T,cyclic):
    supports=[];s=neg
    while True:
        supports.append((s.bit_count(),runs(s,T,cyclic)))
        if s==0:break
        s=(s-1)&neg
    return [(H,D) for H in range(T+1) for D in range(T+1)
            if any(h<=H and d>D for h,d in supports)
            and any(h>H and d<=D for h,d in supports)]

def structures(historical_initial=False):
    result=[]
    for T in range(1,11):
        for neg in range(1<<T):
            rev=int(f'{neg:0{T}b}'[::-1],2)
            pr=[pairs(neg,T,j) for j in range(2)]
            status='search'
            if historical_initial and neg in (0,(1<<T)-1):status='no_charge_or_no_deficit'
            elif neg == 0:status='no_deficit'
            elif not any(pr):status='comparable_for_all_norms'
            elif neg>rev:status='reflection'
            result.append(dict(T=T,negative_mask=neg,signs=['-' if neg&(1<<t) else '+' for t in range(T)],
                reflected_mask=rev,status=status,noncomparable_pairs=[len(p) for p in pr]))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=1150);ap.add_argument('--evals',type=int,default=130)
    ap.add_argument('--historical-initial',action='store_true',help='Replay original search_initial.py; MUST follow with negative_supplement.py')
    args=ap.parse_args();start=time.monotonic();rng=np.random.default_rng(20261022)
    rows=structures(args.historical_initial); (ROOT/'structures.json').write_text(json.dumps(rows,indent=2))
    cfg=dict(seed=20261022,seconds=args.seconds,max_evals_per_structure=args.evals,structures=len(rows),
        searched=sum(r['status']=='search' for r in rows),method='16 fixed/random starts; bounded Powell continuous refinement',
        bounds='negative magnitudes [0,2], positive magnitudes [0,T], E/Pc [0,T]',engine_bisection_iterations=27)
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2));print(json.dumps(cfg),flush=True)
    total=0;best=[None,None];completed=0;trace=open(ROOT/'evaluations.jsonl','w');results=open(ROOT/'results.jsonl','w')
    for row in rows:
        if row['status']!='search':continue
        if time.monotonic()-start>=args.seconds:row['status']='budget_not_started';continue
        T=row['T'];neg=row['negative_mask'];sg=np.array([-1 if neg&(1<<t) else 1 for t in range(T)])
        bounds=[(0.,2. if s<0 else float(T)) for s in sg]+[(0.,float(T))]*2
        localbest=[None,None];count=0;bestp=None;bestscore=-1.
        def objective(p):
            nonlocal total,count,bestp,bestscore
            x=(p[:T]*sg).tolist();ans=evaluate(x,float(p[T]),float(p[T+1]));total+=1;count+=1
            rec=dict(evaluation=total,structure=[T,neg],x=x,E=float(p[T]),Pc=float(p[T+1]),Pd=1,eta=1,results=ans)
            trace.write(json.dumps(rec,separators=(',',':'))+'\n')
            for j,a in enumerate(ans):
                z=dict(rec,selected=a)
                if localbest[j] is None or a['gamma']>localbest[j]['selected']['gamma']+1e-9:localbest[j]=z
                if best[j] is None or a['gamma']>best[j]['selected']['gamma']+1e-9:
                    best[j]=z;(ROOT/'best.json').write_text(json.dumps(best,indent=2))
                    print(json.dumps(dict(event='best',cyclic=j,gamma=a['gamma'],T=T,negative_mask=neg,evaluation=total,x=x,E=rec['E'],Pc=rec['Pc'],H=a['H'],D=a['D'])),flush=True)
            score=max(a['gamma'] for a in ans)
            if score>bestscore+1e-10:bestp=np.array(p);bestscore=score
            return -score
        starts=[]
        for style in range(16):
            p=np.ones(T+2)
            if style==0:p[:T]=np.where(sg<0,.5,1.)
            elif style==1:p[:T]=np.where(sg<0,1.,1.)
            elif style in (2,3):
                vals=np.linspace(.2,1.,sum(sg<0));p[:T][sg<0]=vals if style==2 else vals[::-1]
            elif style<12:
                p[:T][sg<0]=rng.choice([.25,.5,.75,1.,1.25],sum(sg<0));p[T]=rng.choice([.5,1.,1.5,2.,3.]);p[T+1]=rng.choice([.5,1.,2.])
            else:
                p[:T]=rng.uniform(.05,1.5,T);p[T:]=rng.uniform(.1,min(T,4.),2)
            p=np.clip(p,[b[0] for b in bounds],[b[1] for b in bounds]);starts.append(p);objective(p)
        # Powell uses genuinely continuous line searches; it is local and limited.
        opt=minimize(objective,bestp,method='Powell',bounds=bounds,
            options=dict(maxfev=max(1,args.evals-count),maxiter=3,xtol=2e-4,ftol=2e-5))
        row['status']='completed';row['evaluations']=count;row['optimizer_status']=int(opt.status);row['optimizer_message']=str(opt.message)
        row['best']=localbest;results.write(json.dumps(row,separators=(',',':'))+'\n');results.flush();completed+=1
        if completed%25==0:print(json.dumps(dict(event='progress',completed=completed,total=cfg['searched'],evaluations=total,seconds=time.monotonic()-start)),flush=True)
    trace.close();results.close();(ROOT/'structures.json').write_text(json.dumps(rows,indent=2))
    summary=dict(config=cfg,elapsed_seconds=time.monotonic()-start,evaluations=total,completed_search_structures=completed,
        status_counts={k:sum(r['status']==k for r in rows) for k in sorted({r['status'] for r in rows})},best=best,
        all_structures_accounted_for=all(r['status'] not in ('search','budget_not_started') for r in rows),
        continuous_global_optimum_proven=False)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
