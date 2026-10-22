"""Exact piecewise-linear interval inequalities for arbitrary rational examples."""
from fractions import Fraction as F
from pathlib import Path
import json
from search import ROOT,runs

def constraints(x,E,Pc,cover,r):
    T=len(x);q=[max(-x[t]-r,F(0)) if cover&(1<<t) else F(0) for t in range(T)]
    C=[min(Pc,max(a+r,F(0))) for a in x];w=[a-b for a,b in zip(q,C)]
    result=[v-1 for v in q]+[sum(w)]
    for start in range(T):
        cost=F(0)
        for n in range(1,T+1):cost+=w[(start+n-1)%T];result.append(cost-E)
    return result

def threshold(x,E,Pc,cover):
    upper=max([F(0)]+[-a for t,a in enumerate(x) if cover&(1<<t)])
    knots=sorted(set([F(0),upper]+[max(F(0),min(upper,k)) for a in x for k in (-a,Pc-a)]))
    previous=constraints(x,E,Pc,cover,knots[0])
    if max(previous)<=0:return F(0)
    for left,right in zip(knots,knots[1:]):
        current=constraints(x,E,Pc,cover,right)
        if max(current)<=0:
            root=max([left]+[left+(right-left)*a/(a-b) for a,b in zip(previous,current) if a>0])
            assert max(constraints(x,E,Pc,cover,root))<=0
            return root
        previous=current
    raise AssertionError('No feasible upper bound')

def check(name,x,E,Pc,H,D,cyclic):
    x=list(map(F,x));E=F(E);Pc=F(Pc);T=len(x);neg=sum(1<<t for t,a in enumerate(x) if a<0)
    masks=[];cover=neg
    while True:
        loss=neg^cover;h=loss.bit_count();d=runs(loss,T,cyclic);r=threshold(x,E,Pc,cover)
        masks.append(dict(covered_mask=cover,H=h,D=d,r=r,H_ok=h<=H,D_ok=d<=D))
        if not cover:break
        cover=(cover-1)&neg
    rH=min(z['r'] for z in masks if z['H_ok']);rD=min(z['r'] for z in masks if z['D_ok']);rJ=min(z['r'] for z in masks if z['H_ok'] and z['D_ok'])
    return dict(name=name,x=x,E=E,Pc=Pc,Pd=1,H=H,D=D,cyclic=cyclic,r_H=rH,r_D=rD,r_joint=rJ,gamma=rJ-max(rH,rD),supports=masks)

def encode(o):
    if isinstance(o,F):return str(o)
    if isinstance(o,dict):return {k:encode(v) for k,v in o.items()}
    if isinstance(o,list):return [encode(v) for v in o]
    return o

def main():
    cases=[check('linear_best_simplified',[-2,-1,F(-1,2),1,F(-1,2)],1,1,2,1,False),
           check('cyclic_search_best_simplified',[F(-3,4),F(-1,2),-1,F(-5,4),-1,-1,F(-5,4),1,F(-1,2)],F(3,2),1,6,4,True),
           check('cyclic_known',[1,F(-1,2),F(-1,2),-1,-1,-1],1,1,3,2,True),
           check('all_negative_shift',[F(-1,2),-2,-2,F(-5,2),F(-5,2),F(-5,2)],1,1,3,2,True)]
    assert all(c['gamma']==F(1,4) for c in cases)
    out=dict(arithmetic='Fraction, exact roots of piecewise linear interval inequalities',cases=cases)
    (ROOT/'rational_check.json').write_text(json.dumps(encode(out),indent=2));print(json.dumps(encode([{k:v for k,v in c.items() if k!='supports'} for c in cases]),indent=2))

if __name__=='__main__':main()
