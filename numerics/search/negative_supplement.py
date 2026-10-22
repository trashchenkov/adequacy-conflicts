"""Correct the initially unjustified exclusion of entirely negative profiles."""
import json,time
import numpy as np
from scipy.optimize import differential_evolution,minimize
from search import ROOT,evaluate,pairs

start=time.monotonic();seed=2026102203;rows=[];records=[];best=[None,None]
control=evaluate([-.5,-2,-2,-2.5,-2.5,-2.5],1,1)
assert all(abs(a['gamma']-.25)<1e-7 and abs(a['r_joint']-1.75)<1e-7 for a in control)
for T in range(1,11):
    neg=(1<<T)-1;pr=[pairs(neg,T,j) for j in range(2)];row=dict(T=T,negative_mask=neg,noncomparable_pairs=list(map(len,pr)))
    if not any(pr):row['status']='comparable_for_all_norms';rows.append(row);continue
    definitions=[]
    for j in range(2):
        if not pr[j]:definitions.append(dict(cyclic=bool(j),status='comparable'));continue
        local=None;count=0
        def fun(p):
            global count,local
            x=-p[:T];ans=evaluate(x,float(p[T]),float(p[T+1]));count+=1
            rec=dict(T=T,x=x.tolist(),E=float(p[T]),Pc=float(p[T+1]),Pd=1,eta=1,results=ans,selected=ans[j]);records.append(rec)
            if local is None or ans[j]['gamma']>local['selected']['gamma']+1e-9:local=rec
            if best[j] is None or ans[j]['gamma']>best[j]['selected']['gamma']+1e-9:best[j]=rec
            return -ans[j]['gamma']
        bounds=[(0,3)]*T+[(0,T)]*2
        opt=differential_evolution(fun,bounds,maxiter=110,popsize=10,rng=np.random.default_rng(seed+10*T+j),tol=0,atol=-1,polish=False)
        loc=minimize(fun,np.r_[-np.array(local['x']),local['E'],local['Pc']],method='Powell',bounds=bounds,options=dict(maxfev=1000,maxiter=8))
        definitions.append(dict(cyclic=bool(j),status='completed',evaluations=count,best=local,
            differential_status=str(opt.message),local_status=int(loc.status)))
    row.update(status='completed',definitions=definitions);rows.append(row)
    print(json.dumps(dict(T=T,best=[b['selected']['gamma'] if b else None for b in best],seconds=time.monotonic()-start)),flush=True)
out=dict(seed=seed,domain='all x negative with magnitudes[0,3]; E,Pc[0,T]',control_shifted_example=control,rows=rows,best=best,evaluations=len(records),elapsed_seconds=time.monotonic()-start)
(ROOT/'negative_supplement.json').write_text(json.dumps(out,indent=2))
with open(ROOT/'negative_supplement_evaluations.jsonl','w') as f:
    for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('rows','best')}),flush=True)
