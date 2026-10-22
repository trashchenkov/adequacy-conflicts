"""Independent storage checks; no imports from main model.

LP uses explicitly enumerated outage masks and c,d,e_0,...,e_T,
with loss eliminated from the variables. Intended for T<=10.
Large witnesses are checked by direct physical replay, not optimization.
"""
import argparse
import itertools
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent


def max_run(mask, cyclic=False):
    a = np.asarray(mask, bool)
    if not a.size or not a.any():
        return 0
    if a.all():
        return len(a)
    seq = np.tile(a, 2) if cyclic else a
    best = run = 0
    for b in seq:
        run = run + 1 if b else 0
        best = max(best, run)
    return min(best, len(a))


def replay(x, units, r, c, d, e, ell=None, cyclic=True, e0=None,
           initial_fraction=.5, H=None, D=None, ens_limit=None,
           cyclic_duration=False, tol=1e-6):
    """e is post-hour inventory, shape (units,T); checks include limits."""
    x, c, d, e = map(lambda a: np.asarray(a, float), (x, c, d, e))
    m, T = len(units), len(x)
    assert c.shape == d.shape == e.shape == (m, T)
    E = np.array([u['E'] for u in units])[:, None]
    Pc = np.array([u['Pc'] for u in units])[:, None]
    Pd = np.array([u['Pd'] for u in units])[:, None]
    eta = np.array([u['eta'] for u in units])[:, None]
    gamma = np.array([1-u.get('delta', 0) for u in units])[:, None]
    if e0 is None:
        e0 = e[:, -1] if cyclic else initial_fraction*E[:, 0]
    e0 = np.asarray(e0, float)
    previous = np.column_stack((e0, e[:, :-1]))
    residual = e - (gamma*previous + eta*c-d)
    surplus, deficit = np.maximum(x+r, 0), np.maximum(-x-r, 0)
    loss = deficit - d.sum(axis=0)
    positive_max = lambda a: float(max(0., np.max(a)))
    violations = {
        'negative_r': max(0., -float(r)),
        'charge_lower': positive_max(-c), 'charge_upper': positive_max(c-Pc),
        'discharge_lower': positive_max(-d), 'discharge_upper': positive_max(d-Pd),
        'energy_lower': positive_max(-e), 'energy_upper': positive_max(e-E),
        'initial_energy_lower': positive_max(-e0),
        'initial_energy_upper': positive_max(e0-E[:, 0]),
        'aggregate_charge': positive_max(c.sum(axis=0)-surplus),
        'aggregate_discharge': positive_max(d.sum(axis=0)-deficit),
        'dynamics': float(np.max(abs(residual))),
        'boundary': float(np.max(abs(e[:, -1]-e0))) if cyclic else positive_max(e0-e[:, -1]),
    }
    if not cyclic:
        violations['initial_fixed'] = float(np.max(abs(e0-initial_fraction*E[:, 0])))
    if ell is not None:
        violations['reported_loss'] = float(np.max(abs(np.asarray(ell)-loss)))
    ens = float(loss.sum())
    h, run = int(np.sum(loss > tol)), max_run(loss > tol, cyclic_duration)
    if H is not None:
        violations['H'] = max(0, h-H)
    if D is not None:
        violations['D'] = max(0, run-D)
    if ens_limit is not None:
        violations['ENS'] = max(0., ens-ens_limit)
    return {'pass': bool(max(violations.values()) <= tol), 'tolerance': tol,
            'violations': violations, 'ENS': ens, 'H': h, 'D': run,
            'H_at_eps': {str(v): int(np.sum(loss > v)) for v in [1e-9,1e-7,1e-6,1e-5]},
            'D_at_eps': {str(v): max_run(loss > v, cyclic_duration) for v in [1e-9,1e-7,1e-6,1e-5]}}


def mask_lp(x, units, r, outage_mask=None, ens_limit=None,
            cyclic=True, initial_fraction=.5):
    """Minimize ENS with an allowed-outage mask, no indicator variables."""
    x = np.asarray(x, float)
    T, m = len(x), len(units)
    sp, sm = np.maximum(x+r, 0), np.maximum(-x-r, 0)
    n = m*(3*T+1)
    ic = lambda i,t: i*(3*T+1)+t
    idd = lambda i,t: i*(3*T+1)+T+t
    ie = lambda i,t: i*(3*T+1)+2*T+t
    cost = np.zeros(n)
    bounds = [(0,None)]*n
    eq, rhs, ub, rub = [], [], [], []
    for i,u in enumerate(units):
        for t in range(T):
            bounds[ic(i,t)] = (0, min(u['Pc'], sp[t]))
            bounds[idd(i,t)] = (0, min(u['Pd'], sm[t]))
            cost[idd(i,t)] = -1
            row = np.zeros(n)
            row[ie(i,t+1)] = 1
            row[ie(i,t)] = -(1-u.get('delta',0))
            row[ic(i,t)] = -u['eta']
            row[idd(i,t)] = 1
            eq.append(row); rhs.append(0)
        for t in range(T+1):
            bounds[ie(i,t)] = (0,u['E'])
        row = np.zeros(n)
        row[ie(i,0)], row[ie(i,T)] = 1, -1
        if cyclic:
            eq.append(row); rhs.append(0)
        else:
            bounds[ie(i,0)] = (initial_fraction*u['E'], initial_fraction*u['E'])
            ub.append(row); rub.append(0)
    for t in range(T):
        row = np.zeros(n)
        row[[ic(i,t) for i in range(m)]] = 1
        ub.append(row); rub.append(sp[t])
        row = np.zeros(n)
        row[[idd(i,t) for i in range(m)]] = 1
        ub.append(row); rub.append(sm[t])
        if outage_mask is not None and not outage_mask[t]:
            eq.append(row); rhs.append(sm[t])
    if ens_limit is not None:
        ub.append(cost.copy()); rub.append(float(ens_limit-sm.sum()))
    ans = linprog(cost, A_ub=np.array(ub), b_ub=rub,
                  A_eq=np.array(eq), b_eq=rhs, bounds=bounds, method='highs')
    if ans.status == 2:
        return None
    if ans.status != 0:
        raise RuntimeError(ans.message)
    c = np.array([[ans.x[ic(i,t)] for t in range(T)] for i in range(m)])
    d = np.array([[ans.x[idd(i,t)] for t in range(T)] for i in range(m)])
    e = np.array([[ans.x[ie(i,t+1)] for t in range(T)] for i in range(m)])
    return {'ENS': float(sm.sum()+ans.fun), 'c': c, 'd': d, 'e':e}


def enum_feasible(x, units, r, H=None, D=None, ens_limit=None,
                  cyclic_duration=False, cyclic=True):
    T = len(x)
    if H is None and D is None:
        return mask_lp(x, units, r, ens_limit=ens_limit, cyclic=cyclic)
    if T > 12:
        raise ValueError('Enumeration intentionally limited to 12 hours')
    eligible = np.flatnonzero(np.asarray(x)+r < -1e-10)
    for flags in itertools.product([False, True], repeat=len(eligible)):
        mask = np.zeros(T,bool); mask[eligible] = flags
        if H is not None and mask.sum() > H:
            continue
        if D is not None and max_run(mask, cyclic_duration) > D:
            continue
        ans = mask_lp(x, units, r, mask, ens_limit, cyclic)
        if ans is not None:
            return ans
    return None


def bisect_enum(x, units, tol=2e-7, **kwargs):
    if enum_feasible(x, units, 0., **kwargs) is not None:
        return [0.,0.]
    lo, hi = 0., max(0., -min(x))
    while hi-lo > tol:
        mid = (lo+hi)/2
        if enum_feasible(x, units, mid, **kwargs) is None:
            lo = mid
        else:
            hi = mid
    return [lo,hi]


def controls():
    start = time.monotonic()
    U = [{'E':1.,'Pc':1.,'Pd':1.,'eta':1.,'delta':0.}]
    x = [1.,-.5,-.5,-1.,-1.,-1.]
    hd = {k: bisect_enum(x,U,**kw) for k,kw in [('H',{'H':3}),('D',{'D':2}),('HD',{'H':3,'D':2})]}
    assert hd['H'] == hd['D'] == [0.,0.]
    assert abs(hd['HD'][1]-.25) < 2e-6
    U[0]['delta'] = .1
    x = [1.,-1.,-.8]
    sd = {k: bisect_enum(x,U,**kw) for k,kw in [('ENS',{'ens_limit':.9}),('H',{'H':1}),('joint',{'H':1,'ens_limit':.9})]}
    assert sd['ENS'] == sd['H'] == [0.,0.]
    assert abs(sd['joint'][1]-4/95) < 2e-6
    U[0]['delta']=0.
    sd_zero = bisect_enum(x,U,H=1,ens_limit=.9)
    assert sd_zero == [0.,0.]
    rng = np.random.default_rng(20261022)
    common = []
    for k in range(16):
        T = 5 + k%2
        x = rng.uniform(-1.2,1.1,T)
        x[0] = 1.5
        eta = float(rng.uniform(.65,1))
        units = [{'E':float(rng.uniform(.4,1)), 'Pc':.8, 'Pd':.7,'eta':eta,'delta':0.} for _ in range(1+k%2)]
        optimum = mask_lp(x,units,0.)['ENS']
        first = next((h for h in range(T+1) if enum_feasible(x,units,0.,H=h) is not None))
        ans = enum_feasible(x,units,0.,H=first,ens_limit=optimum+1e-7)
        assert ans is not None
        check = replay(x,units,0.,ans['c'],ans['d'],ans['e'],H=first,ens_limit=optimum+1e-7)
        assert check['pass']
        common.append({'T':T,'m':len(units),'min_ENS':optimum,'min_H':first,'joint_feasible_at_zero':True})
    return {'method':'Independent allowed-outage enumeration plus continuous scipy linprog; no main-model imports',
            'duration_exact_quarter':hd, 'self_discharge_exact_4_over_95':sd,
            'zero_self_discharge_reference':sd_zero, 'common_eta_controls':common,
            'runtime_seconds':time.monotonic()-start, 'pass':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--controls', action='store_true')
    args=parser.parse_args()
    if args.controls:
        result={'controls':controls(), 'annual_witnesses':[], 'status':'controls_complete_waiting_for_main_outputs'}
        (HERE/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))



def independent_milp(x, units, r, H=None, D=None, ens_lim=None,
                     time_limit=30., cyclic_duration=False, include_witness=False, fixed_mask=None):
    """Independent full-horizon MILP, ell eliminated, explicit initial state.

    A second formulation with the same HiGHS backend; not a second solver.
    Uses all hours (no compression). Returns statuses separately from replay.
    """
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    x=np.asarray(x,float); T=len(x); m=len(units)
    sp=np.maximum(x+r,0); sm=np.maximum(-x-r,0)
    block=3*T+1; base=m*block; has_mask=H is not None or D is not None or fixed_mask is not None
    n=base+(T if has_mask else 0)
    lb=np.zeros(n); ub=np.full(n,np.inf); cost=np.zeros(n); integrality=np.zeros(n)
    ic=lambda i,t: i*block+t
    idd=lambda i,t: i*block+T+t
    ie=lambda i,t: i*block+2*T+t
    rows=[]; cols=[]; vals=[]; lower=[]; upper=[]
    def add(items,lo=-np.inf,hi=np.inf):
        k=len(lower)
        for col,val in items:
            rows.append(k); cols.append(col); vals.append(val)
        lower.append(lo); upper.append(hi)
    for i,u in enumerate(units):
        for t in range(T):
            ub[ic(i,t)]=min(u['Pc'],sp[t]); ub[idd(i,t)]=min(u['Pd'],sm[t])
            cost[idd(i,t)]=-1
            add([(ie(i,t+1),1),(ie(i,t),-(1-u.get('delta',0))),
                 (ic(i,t),-u['eta']),(idd(i,t),1)],0,0)
        for t in range(T+1): ub[ie(i,t)]=u['E']
        add([(ie(i,0),1),(ie(i,T),-1)],0,0)
    for t in range(T):
        add([(ic(i,t),1) for i in range(m)],hi=sp[t])
        add([(idd(i,t),1) for i in range(m)],hi=sm[t])
        if has_mask:
            ub[base+t]=(1 if sm[t]>0 else 0) if fixed_mask is None else int(bool(fixed_mask[t])); integrality[base+t]=1 if fixed_mask is None else 0
            add([(idd(i,t),1) for i in range(m)]+[(base+t,sm[t])],lo=sm[t])
    if has_mask and H is not None:
        add([(base+t,1) for t in range(T)],hi=H)
    if has_mask and D is not None and D < T:
        for start in range(T if cyclic_duration else T-int(D)):
            add([(base+(start+k)%T,1) for k in range(int(D)+1)],hi=D)
    if ens_lim is not None:
        add([(idd(i,t),1) for i in range(m) for t in range(T)],lo=sm.sum()-ens_lim)
    A=coo_matrix((vals,(rows,cols)),shape=(len(lower),n)).tocsc()
    t0=time.monotonic()
    res=milp(cost,integrality=integrality,bounds=Bounds(lb,ub),
             constraints=LinearConstraint(A,lower,upper),
             options={'time_limit':time_limit,'mip_rel_gap':0})
    out={'status':int(res.status),'message':res.message,'seconds':time.monotonic()-t0,
         'formulation':'explicit initial state; ell eliminated; independent sparse assembly; scipy/HiGHS backend',
         'r':float(r),'H':H,'D':D,'ens_lim':ens_lim}
    if res.x is not None:
        c=np.array([[res.x[ic(i,t)] for t in range(T)] for i in range(m)])
        d=np.array([[res.x[idd(i,t)] for t in range(T)] for i in range(m)])
        e=np.array([[res.x[ie(i,t+1)] for t in range(T)] for i in range(m)])
        e0=np.array([res.x[ie(i,0)] for i in range(m)])
        out['replay']=replay(x,units,r,c,d,e,e0=e0,H=H,D=D,ens_limit=ens_lim,cyclic_duration=cyclic_duration)
        if include_witness and out['replay']['pass']:
            out['witness_arrays']={'c':c.tolist(),'d':d.tolist(),'e':e.tolist(),'e0':e0.tolist()}
    return out


def replay_directory(directory=None):
    directory=Path(directory or HERE/'witnesses')
    out=[]
    for path in sorted(directory.glob('*.npz')):
        with np.load(path,allow_pickle=False) as a:
            units=json.loads(str(a['units_json'].item()))
            limits=json.loads(str(a['limits_json'].item()))
            metadata=json.loads(str(a['metadata_json'].item())) if 'metadata_json' in a else {}
            row=replay(a['x'],units,float(a['r']),a['c'],a['d'],a['e'],a['ell'],
                       H=limits.get('H'),D=limits.get('D'),ens_limit=limits.get('ens_lim'),
                       cyclic=metadata.get('cyclic',True),
                       cyclic_duration=metadata.get('cyclic_duration',False),
                       e0=a['e0'] if 'e0' in a else None)
            if 'allowed_loss' in a:
                mask=np.asarray(a['allowed_loss'])>.5
                loss=np.maximum(-a['x']-float(a['r']),0)-a['d'].sum(axis=0)
                row['allowed_mask_uncovered_residual']=float(max(0,np.max(loss[~mask]))) if (~mask).any() else 0.
                row['pass']=row['pass'] and row['allowed_mask_uncovered_residual'] <= row['tolerance']
            row['file']=path.name
            row['cyclic_duration_diagnostic']=max_run((np.maximum(-a['x']-float(a['r']),0)-a['d'].sum(axis=0))>1e-6,True)
            out.append(row)
    return out


def verify_result_rows(results):
    checks=[]
    for row in results['rows']:
        a,b,j=[row[k] for k in ['r1','r2','rjoint']]
        gamma=[max(0.,j['lo']-max(a['hi'],b['hi'])),
               max(0.,j['hi']-max(a['lo'],b['lo']))]
        report={'id':row['id'], 'gamma_recomputed':gamma,
                'intervals_valid':all(v['lo']<=v['hi']+1e-8 for v in [a,b,j]),
                'gamma_matches':bool(np.max(abs(np.array(gamma)-row['gamma']))<1e-8),
                'certificates':[]}
        for key,lims in [('r1',a['limits']),('r2',b['limits']),('rjoint',row['norms'])]:
            val=row[key]
            if not val.get('witness'):
                report['certificates'].append({'key':key,'witness':None,'basis':val['hi_basis']})
                continue
            with np.load(HERE/val['witness'],allow_pickle=False) as w:
                ck=replay(w['x'],row['units'],float(w['r']),w['c'],w['d'],w['e'],w['ell'],
                          H=lims.get('H'),D=lims.get('D'),ens_limit=lims.get('ens_lim'))
                ck.update(key=key,witness=val['witness'],r_matches_hi=abs(float(w['r'])-val['hi'])<1e-8)
                report['certificates'].append(ck)
        report['pass']=bool(report['intervals_valid'] and report['gamma_matches'] and
                            all(c.get('pass',True) and c.get('r_matches_hi',True) for c in report['certificates']))
        checks.append(report)
    return checks




def exact_single_unit_H(x, unit, r=0., tol=1e-9):
    """Exact-in-model min-H scalar dominance DP, with guaranteed recharge resets.

    At fixed served count only the maximal reachable inventory matters.
    Charge maximally on all surplus hours; at deficit either serve fully or
    do not discharge. Partial discharge cannot improve H or future inventory.
    A recharge reset is certified by propagating a zero lower inventory
    through consecutive nondeficit hours until capacity is reached.
    Numerically solved in floating arithmetic, not a rational certificate.
    """
    start=time.monotonic();x=np.asarray(x,float);T=len(x)
    sp=np.maximum(x+r,0);sm=np.maximum(-x-r,0)
    E,Pc,Pd,eta=map(float,[unit['E'],unit['Pc'],unit['Pd'],unit['eta']])
    gamma=1-float(unit.get('delta',0))
    lower=0.;resets=[]
    # Two passes catch surplus runs crossing the observation boundary.
    for t in range(2*T):
        h=t%T
        lower=0. if sm[h]>0 else min(E,gamma*lower+eta*min(Pc,sp[h]))
        if t>=T and lower>=E-tol:resets.append(h)
    if not resets:
        return {'established':False,'reason':'no guaranteed-full recharge reset found',
                'runtime_seconds':time.monotonic()-start}
    anchor=resets[0];order=(np.arange(T)+anchor+1)%T;resetset=set(resets)
    segments=[];segment=[]
    for h in order:
        segment.append(int(h))
        if int(h) in resetset:
            segments.append(segment);segment=[]
    assert not segment and order[-1]==anchor
    serve=np.zeros(T,bool);summary=[]
    for seq in segments:
        states=np.array([E]);parents=[]
        for h in seq:
            before=gamma*states
            if sm[h]<=0:
                states=np.minimum(E,before+eta*min(Pc,sp[h]))
                parents.append(np.zeros(len(states),bool))
            else:
                after=np.r_[before,-np.inf];parent=np.zeros(len(after),bool)
                if sm[h]<=Pd+tol:
                    candidate=before-sm[h]
                    eligible=candidate>=-tol
                    candidate=np.where(eligible,np.maximum(candidate,0),-np.inf)
                    improve=candidate>after[1:]+tol
                    after[1:]=np.maximum(after[1:],candidate)
                    parent[1:]=improve
                # Drop impossible terminal counts; interior -inf is harmless.
                last=np.flatnonzero(np.isfinite(after))[-1]+1
                states=after[:last];parents.append(parent[:last])
        best=len(states)-1
        assert abs(states[best]-E)<=1e-6
        k=best
        for h,parent in reversed(list(zip(seq,parents))):
            if parent[k]:serve[h]=True;k-=1
        assert k==0
        summary.append({'hours':len(seq),'deficit_hours':int(np.sum(sm[seq]>0)),
                        'served':best,'ending_hour':seq[-1]})
    c=np.zeros((1,T));d=np.zeros((1,T));e=np.zeros((1,T));energy=E
    for h in order:
        before=gamma*energy
        if sp[h]>0:
            c[0,h]=min(Pc,sp[h],max(0,(E-before)/eta))
        if serve[h]:d[0,h]=sm[h]
        energy=before+eta*c[0,h]-d[0,h];e[0,h]=energy
    H=int(np.sum((sm>0)&~serve))
    ck=replay(x,[unit],r,c,d,e,H=H)
    assert ck['pass'],ck
    return {'established':True,'min_H':H,'served_deficit_hours':int(serve.sum()),
            'deficit_hours':int(np.sum(sm>0)), 'reset_count':len(resets),
            'max_segment_hours':max(map(len,segments)), 'segments':summary,
            'runtime_seconds':time.monotonic()-start,'replay':ck,
            'c':c,'d':d,'e':e,'serve':serve,
            'method':'Scalar maximum-SOC dominance DP; independently certified full-recharge resets; float arithmetic'}


if __name__ == '__main__':
    main()
