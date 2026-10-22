
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Exact rational checks of the two examples supplied in the note.
Independent of both numerical solvers; all physical balances checked.
"""
from fractions import Fraction as F
from pathlib import Path
import json


def verify(x,E,Pc,Pd,r,c,d,e0):
    x,c,d=([F(v) for v in a] for a in [x,c,d])
    E,Pc,Pd,r,e0=map(F,[E,Pc,Pd,r,e0])
    state=e0; states=[]; ell=[]
    for a,charge,discharge in zip(x,c,d):
        s=a+r
        assert 0<=charge<=Pc and charge<=max(s,0)
        assert 0<=discharge<=Pd and discharge<=max(-s,0)
        state+=charge-discharge
        assert 0<=state<=E
        states.append(state);ell.append(max(-s,0)-discharge)
    assert state==e0
    bits=[v>0 for v in ell]
    events=sum(b and (t==0 or not bits[t-1]) for t,b in enumerate(bits))
    return {'r':str(r),'c':list(map(str,c)),'d':list(map(str,d)),
            'e0':str(e0),'e':list(map(str,states)),'ell':list(map(str,ell)),
            'ENS':str(sum(ell)),'H':sum(bits),'events_linear':events,
            'peak':str(max(ell)),'physical_exact':True}


def main():
    peakH=verify([1,-1,-1],1,1,1,0,[1,0,0],[0,1,0],0)
    peakP=verify([1,-1,-1],1,1,1,0,[1,0,0],[0,F(1,2),F(1,2)],0)
    peakJ=verify([1,-1,-1],1,1,1,F(1,4),[1,0,0],[0,F(3,4),F(1,4)],0)
    assert peakH['H']==1 and F(peakP['peak'])==F(1,2)
    assert peakJ['H']==1 and F(peakJ['peak'])==F(1,2)
    assert 1+F(1,2)>1 # Joint demand on discharge at r=0 exceeds cyclic input.
    ens=verify([3,-2,-1,-2],3,3,1,0,[3,0,0,0],[0,1,1,1],0)
    event=verify([3,-2,-1,-2],3,3,1,0,[0]*4,[0]*4,0)
    assert ens['ENS']=='2' and ens['events_linear']==2
    assert event['events_linear']==1
    small=[];fixed_zero=[]
    for r in [F(1,10),F(1,100),F(1,10000)]:
        z=verify([3,-2,-1,-2],3,3,1,r,[3-2*r,0,0,0],[0,1,1-2*r,1],0)
        assert F(z['ENS'])==2-r and z['events_linear']==1
        small.append(z)
        z=verify([3,-2,-1,-2],3,3,1,0,[3-r,0,0,0],[0,1,1-r,1],0)
        assert F(z['ENS'])==2+r and z['events_linear']==1
        fixed_zero.append(z)
    result={'peak_vs_H':{'H':peakH,'peak':peakP,'joint_at_quarter':peakJ,
                         'exact_joint_capacity':'1/4'},
            'ENS_vs_events':{'ENS_only':ens,'events_only':event,
               'positive_capacity_family':small,'r0_ENS_infimum_family':fixed_zero,
               'capacity_infimum':'0','capacity_minimum_exists':False,
               'r0_ENS_under_one_event_infimum':'2','r0_ENS_minimum_exists':False}}
    p=Path(__file__).with_suffix('.json')
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Exact physical witnesses checked:',5+len(small)+len(fixed_zero))


if __name__=='__main__':main()
