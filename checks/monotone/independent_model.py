
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Independent balance model, written without reading/refactoring ref_milp.py.

Fixed source output gives known external-surplus/deficit signs. At delta=0,
full-output normalization justifies using output=r for monotone objectives.
All units use freely selected cyclic SOC. Powers MW, energy MWh, h=1.
"""
from dataclasses import dataclass, asdict
import warnings
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, vstack


@dataclass
class Case:
    x: list
    E: list
    Pc: list
    Pd: list
    eta: list

    def record(self):
        return asdict(self)


CALLS = []


def runs(mask, cyclic=False):
    bits = list(map(bool, mask))
    if not any(bits):
        return 0
    if all(bits):
        return len(bits)
    seq = bits + bits if cyclic else bits
    cur = best = 0
    for b in seq:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def dispatch(case, r, *, output=None, shortage_cap=None, ens_cap=None,
             hours_cap=None, run_cap=None, cyclic=False, objective=None,
             shortage_floor=None, label=''):
    x = np.asarray(case.x, float)
    E, pc, pd, eta = map(lambda a: np.asarray(a, float),
                         (case.E, case.Pc, case.Pd, case.eta))
    n, T = len(E), len(x)
    f = np.full(T, r) if output is None else np.asarray(output, float)
    if np.any(f < -1e-12) or np.any(f > r + 1e-12):
        raise ValueError('source output outside capacity')
    S, D = np.maximum(x + f, 0), np.maximum(-x - f, 0)
    q = n*T
    c = np.arange(q).reshape(n, T)
    d, e = c + q, c + 2*q
    ell = np.arange(3*q, 3*q+T)
    discrete = hours_cap is not None or run_cap is not None
    z = np.arange(3*q+T, 3*q+2*T) if discrete else None
    nv = 3*q + T*(2 if discrete else 1)
    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    ub[c.ravel()] = (pc[:, None]*(S[None, :] > 0)).ravel()
    ub[d.ravel()] = (pd[:, None]*(D[None, :] > 0)).ravel()
    ub[e.ravel()] = np.broadcast_to(E[:, None], (n, T)).ravel()
    ub[ell] = D
    if shortage_cap is not None:
        ub[ell] = np.minimum(D, np.maximum(shortage_cap, 0))
    if shortage_floor is not None:
        lb[ell] = shortage_floor
    if discrete:
        ub[z] = (D > 0).astype(float)
    eq, beq, iq, biq = [], [], [], []

    def row():
        return np.zeros(nv)

    for i in range(n):
        for t in range(T):
            a = row()
            a[e[i,t]] += 1
            a[e[i,(t-1) % T]] -= 1
            a[c[i,t]], a[d[i,t]] = -eta[i], 1
            eq.append(a); beq.append(0.)
    for t in range(T):
        a = row(); a[d[:,t]] = 1; a[ell[t]] = 1
        eq.append(a); beq.append(D[t])
        a = row(); a[c[:,t]] = 1
        iq.append(a); biq.append(S[t])
    if ens_cap is not None:
        a = row(); a[ell] = 1
        iq.append(a); biq.append(ens_cap)
    if discrete:
        for t in range(T):
            a = row(); a[ell[t]] = 1; a[z[t]] = -D[t]
            iq.append(a); biq.append(0.)
        if hours_cap is not None:
            a = row(); a[z] = 1
            iq.append(a); biq.append(int(np.floor(hours_cap)))
        if run_cap is not None:
            m = int(np.floor(run_cap))
            if m < T:
                for t in range(T if cyclic else T-m):
                    a = row(); a[z[(np.arange(m+1)+t) % T]] = 1
                    iq.append(a); biq.append(m)
    A, b = csr_matrix(np.asarray(eq)), np.asarray(beq)
    U, v = csr_matrix(np.asarray(iq)), np.asarray(biq)
    obj = np.zeros(nv)
    if objective is None:
        obj[ell] = 1
    else:
        obj[:3*q+T] = objective
    if discrete:
        integ = np.zeros(nv); integ[z] = 1
        # These options belong to bundled HiGHS (checked via HighsOptions),
        # and scipy forwards them despite its narrower public option list.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*',
                                    category=RuntimeWarning)
            res = milp(obj, integrality=integ, bounds=Bounds(lb, ub),
                       constraints=LinearConstraint(vstack([A,U]),
                         np.r_[b,np.full(len(v),-np.inf)], np.r_[b,v]),
                       options={'time_limit':10., 'mip_rel_gap':0.,
                                'mip_feasibility_tolerance':1e-9,
                                'primal_feasibility_tolerance':1e-9,
                                'dual_feasibility_tolerance':1e-9})
    else:
        res = linprog(obj, A_ub=U, b_ub=v, A_eq=A, b_eq=b,
                      bounds=list(zip(lb,ub)), method='highs',
                      options={'primal_feasibility_tolerance':1e-9,
                               'dual_feasibility_tolerance':1e-9})
    out = {'id':len(CALLS), 'label':label, 'status':int(res.status),
           'message':str(res.message), 'r':float(r),
           'kind':'MILP' if discrete else 'LP',
           'gap':float(getattr(res,'mip_gap',0.)) if res.status==0 else None}
    if res.status == 0:
        sol = res.x
        violation = max(float(np.max(np.abs(A@sol-b))),
                        float(np.max(np.maximum(U@sol-v,0))),
                        float(np.max(np.maximum(lb-sol,0))),
                        float(np.max(np.maximum(sol-ub,0))))
        l = sol[ell]
        out.update(ens=float(sum(l)), ell=l.tolist(), c=sol[c].tolist(),
                   d=sol[d].tolist(), e=sol[e].tolist(), f=f.tolist(),
                   violation=violation, hours=int(np.sum(l>1e-7)),
                   longest=runs(l>1e-7,cyclic), objective=float(res.fun))
        if discrete:
            out.update(z=sol[z].tolist(),
                       integer_violation=float(np.max(np.abs(sol[z]-np.rint(sol[z])))))
    CALLS.append(out)
    return out


def bound_B(case):
    eta=np.asarray(case.eta); w=1-eta/max(eta)
    D=np.minimum(case.Pd,case.E)
    C=np.minimum(case.Pc,np.asarray(case.E)/eta)
    n=len(w)
    obj=np.r_[np.zeros(n),1.]
    U=np.vstack([np.r_[D,-1.],np.r_[-C,-1.]])
    res=linprog(obj,A_ub=U,b_ub=[0.,-float(w@C)],
                bounds=[(0,float(a)) for a in w]+[(0,None)],method='highs')
    return {'status':int(res.status),'B':float(res.fun) if res.status==0 else None,
            'u':res.x[:-1].tolist() if res.status==0 else None,
            'Delta':float(w@np.asarray(case.Pd)), 'D':D.tolist(),'C':C.tolist()}


def capacity(case, *, eps=1e-6, label='', **limits):
    start=len(CALLS)
    a=dispatch(case,0.,label=label+':zero',**limits)
    if a['status']==0:
        return {'lo':0.,'hi':0.,'status':'optimal_zero','solution':a,
                'calls':[start,len(CALLS)]}
    if a['status']!=2:
        return {'status':'unknown','failed':a,'calls':[start,len(CALLS)]}
    lo,hi=0.,float(max(0,-min(case.x)))
    best=dispatch(case,hi,label=label+':upper',**limits)
    if best['status']!=0:
        return {'status':'unknown','failed':best,'calls':[start,len(CALLS)]}
    while hi-lo>eps:
        mid=(lo+hi)/2
        cur=dispatch(case,mid,label=label+':bisect',**limits)
        if cur['status']==0:
            hi,best=mid,cur
        elif cur['status']==2:
            lo=mid
        else:
            return {'status':'unknown','failed':cur,'calls':[start,len(CALLS)]}
    return {'lo':lo,'hi':hi,'status':'bracket','solution':best,
            'calls':[start,len(CALLS)]}
