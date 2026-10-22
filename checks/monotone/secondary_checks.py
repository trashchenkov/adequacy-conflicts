
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Cross-check the finished independent implementation against ref_milp,
and compare duration MILP with explicit allowed-support LP enumeration.
No source code is copied from the reference model.
"""
from pathlib import Path
import itertools,json
import numpy as np
from independent_model import Case,dispatch,capacity,bound_B,runs,CALLS
import ref_milp

ROOT=Path(__file__).resolve().parent


def units(case):
    return [dict(E=e,Pc=c,Pd=d,eta=a,delta=0.)
            for e,c,d,a in zip(case.E,case.Pc,case.Pd,case.eta)]


def main():
    equal=json.loads((ROOT/'results/equal.json').read_text())
    unequal=json.loads((ROOT/'results/unequal.json').read_text())
    cross=[]
    for k,row in enumerate(equal[:25]+unequal[:25]):
        case=Case(**row['case']);r=row['r'];H=len(case.x)//2
        for cap in [None,H]:
            ours=dispatch(case,r,hours_cap=cap,label=f'reference:{k}:{cap}')
            try:
                theirs=ref_milp.solve(case.x,units(case),r,H=cap)
                good=(theirs is None and ours['status']==2) or (
                    theirs is not None and ours['status']==0 and
                    abs(theirs-ours['ens'])<1e-7*max(1,abs(theirs)))
                cross.append({'case':case.record(),'r':r,'H':cap,
                              'our_solution':ours,'reference_ens':theirs,'match':good})
            except RuntimeError as e:
                cross.append({'case':case.record(),'r':r,'H':cap,'unknown':str(e)})
    support=[]
    short=[a for a in unequal if len(a['case']['x'])<=7][:10]
    for k,row in enumerate(short):
        case=Case(**row['case']);r=row['r'];T=len(case.x)
        external=np.maximum(-np.array(case.x)-r,0)
        for cyclic in [False,True]:
            D=1+(k%2);H=T//2
            mixed=dispatch(case,r,hours_cap=H,run_cap=D,cyclic=cyclic,
                           label=f'support:{k}:{cyclic}:milp')
            allsol=[]
            for mask in itertools.product([0,1],repeat=T):
                if sum(mask)>H or runs(mask,cyclic)>D:continue
                sol=dispatch(case,r,shortage_cap=external*mask,
                             label=f'support:{k}:{cyclic}:lp')
                allsol.append(sol)
            ok=[s for s in allsol if s['status']==0]
            unknown=[s for s in allsol if s['status'] not in (0,2)]
            best=min((s['ens'] for s in ok),default=None)
            good=(best is None and mixed['status']==2) or (
                best is not None and mixed['status']==0 and abs(best-mixed['ens'])<1e-7)
            support.append({'case':case.record(),'r':r,'H':H,'D':D,'cyclic':cyclic,
                            'milp':mixed,'support_lp_count':len(allsol),'lp_min_ens':best,
                            'match':good,'unknown':unknown})
    examples=[]
    for kind,case,H,D,cyclic,exact in [
        ('linear',Case([-1,2,-1,-2,-2],[2],[2],[2],[1]),2,1,False,.5),
        ('cyclic',Case([2,-1,-1,-2,-2,-2],[2],[2],[2],[1]),3,2,True,.5)]:
        rh=capacity(case,hours_cap=H,label='duration:'+kind+':H')
        rd=capacity(case,run_cap=D,cyclic=cyclic,label='duration:'+kind+':D')
        rj=capacity(case,hours_cap=H,run_cap=D,cyclic=cyclic,label='duration:'+kind+':J')
        attained=dispatch(case,exact,hours_cap=H,run_cap=D,cyclic=cyclic,
                          label='duration:'+kind+':exact')
        below=dispatch(case,exact-1e-5,hours_cap=H,run_cap=D,cyclic=cyclic,
                       label='duration:'+kind+':below')
        examples.append({'kind':kind,'case':case.record(),'H':H,'D':D,
                         'cyclic':cyclic,'exact_r':exact,'rH':rh,'rD':rd,'rJ':rj,
                         'exact_solution':attained,'below_solution':below})
        assert attained['status']==0 and below['status']==2
    known=Case([1,1,-1.5,-3],[2,2],[1,1],[1,1],[.8,.95])
    knownB={'case':known.record(),'bound':bound_B(known),'exact_gamma':'13/654',
            'rE':capacity(known,ens_cap=209/80,label='Bcontrol:E'),
            'rD':capacity(known,run_cap=1,label='Bcontrol:D'),
            'rJ':capacity(known,ens_cap=209/80,run_cap=1,label='Bcontrol:J')}
    out={'reference_comparisons':cross,'support_comparisons':support,
         'duration_examples':examples,'positive_B_control':knownB,
         'summary':{'reference_n':len(cross),'reference_matches':sum(a.get('match',False) for a in cross),
                    'support_n':len(support),'support_matches':sum(a['match'] for a in support),
                    'support_lp_n':sum(a['support_lp_count'] for a in support),
                    'max_physical_violation':max(a.get('violation',0) for a in CALLS)}}
    (ROOT/'results/secondary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'results/secondary_calls.json').write_text(json.dumps(CALLS,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out['summary'],indent=2))
    assert all(a.get('match',False) for a in cross)
    assert all(a['match'] and not a['unknown'] for a in support)


if __name__=='__main__':main()
