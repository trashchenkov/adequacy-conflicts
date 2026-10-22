"""LP cross-check of the interval oracle, including all-cover threshold tests."""
import json
import numpy as np
from scipy.optimize import linprog
from search import evaluate,ROOT

rng=np.random.default_rng(62841);records=[]
for case in range(80):
    T=int(rng.integers(2,11));x=rng.uniform(-2,2,T);E=float(rng.uniform(0,4));Pc=float(rng.uniform(0,3))
    ans,tab=evaluate(x,E,Pc,True);valid=np.flatnonzero(np.isfinite(tab))
    for cover in rng.choice(valid,min(6,len(valid)),replace=False):
        threshold=tab[cover]
        for r in [threshold+1e-6,max(0,threshold-1e-6)]:
            q=np.array([max(-x[t]-r,0) if cover&(1<<t) else 0 for t in range(T)])
            C=np.minimum(Pc,np.maximum(x+r,0));A=np.zeros((T,2*T))
            for t in range(T):A[t,t]=1;A[t,(t-1)%T]-=1;A[t,T+t]=-1
            lp=linprog(np.zeros(2*T),A_eq=A,b_eq=-q,bounds=[(0,E)]*T+[(0,c) for c in C],method='highs')
            ok=lp.status==0 and np.max(q)<=1+1e-9
            expected=r>=threshold-1e-8
            records.append(dict(case=case,T=T,cover=int(cover),r=r,threshold=float(threshold),lp_status=int(lp.status),power_ok=bool(np.max(q)<=1+1e-9),match=bool(ok==expected)))
assert all(r['match'] for r in records)
known=evaluate([1,-.5,-.5,-1,-1,-1],1,1)
assert all(abs(a['gamma']-.25)<1e-8 for a in known)
out=dict(seed=62841,random_cases=80,checks=len(records),failures=sum(not r['match'] for r in records),known_example=known,records=records)
(ROOT/'engine_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='records'}))
