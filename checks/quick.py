"""Numerical checks against manuscript constants, not a proof of the theorems."""
from pathlib import Path
import sys
import json
import math
from fractions import Fraction as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'figures')]
import ref_milp as ref
from make_figures import competition_gamma, DAY, shore_gamma

ROWS=[]
def check(name,expected,obtained,tol=2e-6):
    ok=math.isfinite(obtained) and abs(obtained-expected)<=tol
    ROWS.append(dict(name=name,expected=expected,obtained=obtained,tolerance=tol,ok=ok))
    print(f'{"ok" if ok else "FAIL"} {name}: expected={expected:.12g}, obtained={obtained:.12g}, tol={tol:g}',flush=True)
    return ok

def gamma(name,x,U,H,D,target):
    c=dict(x=x,units=U)
    a=ref.bisect_hd(c,H,None,tol=2e-7)
    b=ref.bisect_hd(c,None,D,tol=2e-7)
    j=ref.bisect_hd(c,H,D,tol=2e-7)
    check(name,target,j[1]-max(a[1],b[1]))

def run():
    U=[dict(E=1,Pc=1,Pd=1,eta=1,delta=0)]
    gamma('ex:quarter Gamma',[1,-.5,-.5,-1,-1,-1],U,3,2,.25)
    # thm:half sharpness concerns TWO PINNED HOURS, not the H/D example in hd2_check.py.
    # Exact individual witnesses (one unit discharged at either hour) at r=0.
    assert sum([F(1),-F(1),F(0)])==0
    assert sum([F(1),F(0),-F(1)])==0
    both=ref.bisect_hd(dict(x=[1,-1,-1],units=U),0,None,tol=2e-7)
    check('thm:half pinned-hour sharpness',.5,both[1])
    examples=[('ex:sd 4/95',[1,-1,-.8],1,1,1,.1,4/95),
              ('prop:three capacity',[1,-1,-.2],.5,1,1,.1,.2*.1/1.9),
              ('prop:three surplus eta=1',[1,-2,-.5],10,10,1,.1,.5*.1/(1+.9+.9**2)),
              ('prop:three surplus eta=.5',[1,-2,-.2],10,10,.5,.1,.2*.1/(1+.9+.5*.9**2))]
    for name,x,E,Pc,eta,delta,target in examples:
        z=competition_gamma(x,E,Pc,1,eta,delta,tol=2e-7)
        check(name,target,(z['lo']+z['hi'])/2)
    for delta,L in [(.1,1),(.05,3),(.02,10),(.01,20)]:
        g=1-delta;x=[1.]+[-3.]*L+[-g**(L+1)]
        z=competition_gamma(x,1,1,1,1,delta,tol=2e-7)
        check(f'prop:shore delta={delta} L={L}',float(shore_gamma(delta,L)),(z['lo']+z['hi'])/2)
    z=competition_gamma(DAY,3,1,1,1,.01)
    check('daily tank delta=.01 Gamma',.006687,z['hi'],2e-5)
    # Replay frozen pre-adaptation base-vs-g6 regression against the only solver copy.
    baseline=json.loads((ROOT/'validation/solver_equivalence.json').read_text())
    matches=0
    for q in baseline['rows']:
        value=ref.solve(q['x'],q['units'],q['r'],**q['kwargs'])
        expected=q['base']
        matches+=int((value is None and expected is None) or (value is not None and expected is not None and abs(value-expected)<1e-7))
    check('MILP no-D regression',baseline['count'],matches,0)
    (ROOT/'validation/quick_theory.json').write_text(json.dumps(ROWS,indent=2)+'\n')
    if not all(q['ok'] for q in ROWS):raise SystemExit(1)

if __name__=='__main__':run()
