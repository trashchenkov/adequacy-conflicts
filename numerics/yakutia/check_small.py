"""Cross-check sparse MILP and necessary bounds against prior independent code."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import importlib.util,sys,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
new=load('duration_sparse',HERE/'run.py')
old=load('duration_prior_independent',HERE.parents[1]/'checks/monotone/independent_model.py')
rng=np.random.default_rng(20261022);rows=[]
for j in range(30):
    n=1+j%2;T=8+j%3
    x=rng.uniform(-3,2,T)
    e=dict(E=rng.uniform(.8,3,n).tolist(),Pc=rng.uniform(.4,2,n).tolist(),Pd=rng.uniform(.4,2,n).tolist(),eta=rng.uniform(.3,.95,n).tolist())
    r=float(rng.uniform(0,1.8));H=int(rng.integers(0,T));D=int(rng.integers(1,T-1))
    a,w=new.solve(x,e,r,H,D,limit=2.)
    z=old.dispatch(old.Case(x.tolist(),**e),r,hours_cap=H,run_cap=D,cyclic=True)
    passed=(a['accepted']==(z['status']==0)) and (not a['accepted'] or new.necessary(x,e,r,H,D))
    rows.append({'j':j,'new_status':a['status'],'old_status':z['status'],'necessary':new.necessary(x,e,r,H,D),'passed':bool(passed),'parameters':dict(x=x.tolist(),r=r,H=H,D=D,**e)})
assert all(a['passed'] for a in rows)
(HERE/'small_independent_validation.json').write_text(json.dumps({'passed':True,'count':len(rows),'checks':rows},indent=2))
print('PASS',len(rows),'independent small MILP comparisons and necessary-condition checks')
