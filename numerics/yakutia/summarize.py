"""Independent witness audit and interval-safe catalogue comparison."""
import json,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]/"data/yakutia"

def longest(bits):
    n=len(bits)
    if all(bits):return n
    q=[i for i,v in enumerate(bits) if not v]
    return max((q[(j+1)%len(q)]-q[j]-1)%n for j in range(len(q)))

def main():
    raw=json.loads((HERE/'results.json').read_text());rows=raw['rows']
    equipment={a['id']:a for a in json.loads((BASE/'online_pilot_2026-10-16/equipment.json').read_text())['cases']}
    profiles={}
    for file,tag in [('annual_8760.npz','annual'),('winter_1008.npz','winter')]:
        with np.load(BASE/'online_pilot_2026-10-16/data'/file) as p:
            for year,x in zip(p['years'],p['x_model_kW']):profiles[f'{tag}{year}']=x
    p=json.loads((BASE/'five_storage_parameter_grid/cases.json').read_text())['profiles']
    for k,v in p.items():profiles['pilot72_'+k]=.4*np.array(v['generation_R1'])-np.array(v['load'])
    checked={};failures=[];maximum=0.;max_unmarked_loss=0.
    for a in rows:
        e=equipment[a['equipment']];x=profiles[a['profile']]-e['aux_load_kw'];eq=e['aggregated']
        for key in ('r_H','r_D','r_joint'):
            b=a[key]
            if b['lo']>b['hi']+1e-8:failures.append(['inconsistent_interval',a['id'],key,b['lo'],b['hi']])
            if not b.get('witness'):continue
            path=HERE/b['witness']
            with np.load(path) as w:
                c,d,s,ell,allowed,r=[w[k] for k in ('c','d','soc','ell','allowed_loss','r')]
                r=float(r);S=np.maximum(x+r,0);Q=np.maximum(-x-r,0)
                errs=[np.max(np.abs(s-np.roll(s,1,axis=1)-np.asarray(eq['eta'])[:,None]*c+d)),np.max(np.abs(d.sum(axis=0)+ell-Q)),max(0,np.max(c.sum(axis=0)-S)),max(0,np.max(s-np.asarray(eq['E'])[:,None])),max(0,np.max(c-np.asarray(eq['Pc'])[:,None])),max(0,np.max(d-np.asarray(eq['Pd'])[:,None])),max(0,-np.min(s)),max(0,-np.min(c)),max(0,-np.min(d)),max(0,-np.min(ell)),float(np.max(ell[~allowed])) if np.any(~allowed) else 0.]
                maximum=max(maximum,max(errs));h=int(sum(allowed));dc=longest(allowed)
                max_unmarked_loss=max(max_unmarked_loss,errs[-1])
                good=max(errs)<1e-8 and (key=='r_D' or h<=a['H']) and (key=='r_H' or dc<=a['D'])
                if not good:failures.append([a['id'],key,errs,h,dc])
                checked[str(path.name)]={'max_residual_kW_or_kWh':max(errs),'max_ell_outside_allowed_loss_kW':errs[-1],'hours':h,'longest_cyclic':dc,'r':r,'profile':a['profile'],'equipment':a['equipment']}
    # Each feasible joint witness is also feasible for either separate constraint.
    # Sharing only bounds whose inequalities imply the target constraints.
    for a in rows:
        for key in ('r_H','r_D','r_joint'):
            b=a[key];b['raw_lo']=b['lo'];b['raw_hi']=b['hi']
            b['lo_provenance']={'row':a['id'],'component':key,'value':b['lo'],'kind':b['lo_source']}
            b['hi_provenance']={'row':a['id'],'component':key,'value':b['hi'],'kind':b['hi_source'],'witness':b.get('witness')}
    def upper(target,source):
        if source['hi']<target['hi']:
            target['hi']=source['hi'];target['hi_provenance']=dict(source['hi_provenance'])
    def lower(target,source):
        if source['lo']>target['lo']:
            target['lo']=source['lo'];target['lo_provenance']=dict(source['lo_provenance'])
    for a in rows:
        for name,w in checked.items():
            if (a['profile'],a['equipment'])!=(w['profile'],w['equipment']):continue
            source={'hi':w['r'],'hi_provenance':{'kind':'independently audited support meets target limits','witness':'witnesses/'+name,'value':w['r'],'hours':w['hours'],'longest_cyclic':w['longest_cyclic']}}
            if w['hours']<=a['H']:upper(a['r_H'],source)
            if w['longest_cyclic']<=a['D']:upper(a['r_D'],source)
            if w['hours']<=a['H'] and w['longest_cyclic']<=a['D']:upper(a['r_joint'],source)
    for _ in range(3):
        for target in rows:
            peers=[a for a in rows if a['profile']==target['profile'] and a['equipment']==target['equipment']]
            for src in peers:
                if src['H']<=target['H']:
                    upper(target['r_H'],src['r_H']);upper(target['r_H'],src['r_joint'])
                if src['D']<=target['D']:
                    upper(target['r_D'],src['r_D']);upper(target['r_D'],src['r_joint'])
                if src['H']>=target['H']:lower(target['r_H'],src['r_H'])
                if src['D']>=target['D']:lower(target['r_D'],src['r_D'])
                if src['H']<=target['H'] and src['D']<=target['D']:upper(target['r_joint'],src['r_joint'])
                if src['H']>=target['H'] and src['D']>=target['D']:lower(target['r_joint'],src['r_joint'])
            lower(target['r_joint'],target['r_H']);lower(target['r_joint'],target['r_D'])
            if target['structural_zero']:
                lower(target['r_H'],target['r_joint']);upper(target['r_joint'],target['r_H'])
    catalogue=json.loads((BASE/'catalog_significance_2026-10-14/summary.json').read_text())['catalogues']
    catrows=[]
    for a in rows:
        for key in ('r_H','r_D','r_joint'):
            b=a[key]
            if b['hi']!=b['raw_hi']:
                b['hi_source']='monotone implication from same profile/equipment; compare raw results'
                if 'witness' in b:b['original_witness']=b.pop('witness')
            if b['lo']!=b['raw_lo']:b['lo_source']='monotone implication from same profile/equipment; compare raw results'
        h,d,j=a['r_H'],a['r_D'],a['r_joint'];L0=max(h['lo'],d['lo']);U0=min(j['hi'],max(h['hi'],d['hi']))
        a['r0']=[L0,U0]
        a['gamma']=[0.,0.] if a['structural_zero'] else [max(0.,j['lo']-U0),max(0.,j['hi']-L0)]
        if a['gamma'][0]>a['gamma'][1]+1e-8:failures.append(['inconsistent_Gamma_interval',a['id'],a['gamma']])
        a['gamma_fraction_Pd']=[v/a['sum_Pd'] for v in a['gamma']]
        for cat in catalogue:
            levels=np.array(cat['ratings_kw']);ceiling=lambda v:float(levels[np.searchsorted(levels,v-1e-7)])
            r0nom=[ceiling(L0),ceiling(U0)];jnom=[ceiling(j['lo']),ceiling(j['hi'])]
            # Bounds for positive local rating increments across the whole r0 interval.
            ixlo=np.searchsorted(levels,L0);ixhi=np.searchsorted(levels,U0)
            steps=[float(levels[i]-levels[i-1]) for i in range(max(1,ixlo),max(1,ixhi)+1)]
            same=a['structural_zero'] or (r0nom[0]==r0nom[1]==jnom[0]==jnom[1])
            catrows.append({'id':a['id'],'catalog_id':cat['catalog_id'],'mode':'single','r0_nominal_interval':r0nom,'joint_nominal_interval':jnom,'nominal_increment_interval':[0. if same else max(0.,jnom[0]-r0nom[1]),0. if same else max(0.,jnom[1]-r0nom[0])],'status':'same_certified' if same else 'unresolved','local_step_interval':list((min(steps),max(steps))),'gamma_over_step_interval':[a['gamma'][0]/max(steps),a['gamma'][1]/min(steps)]})
    verification={'passed':not failures,'witness_files_checked':len(checked),'max_residual':maximum,'max_ell_outside_allowed_loss_kW':max_unmarked_loss,'max_integer_violation':max((a.get('integer_violation',0) for a in raw['calls']),default=0),'failures':failures,'witnesses':checked,'note':'independent direct balances at tolerance1e-8; no rounded-zero Gamma for annual intervals'}
    (HERE/'verification.json').write_text(json.dumps(verification,indent=2))
    (HERE/'catalogues.json').write_text(json.dumps(catrows,indent=2))
    (HERE/'derived_results.json').write_text(json.dumps({'rows':rows,'interval_propagation':'monotone subset implications; original solver bounds and witnesses remain in results.json'},ensure_ascii=False,indent=2))
    with (HERE/'table.csv').open('w') as f:
        fields=['id','T','H','D','rH_lo','rH_hi','rD_lo','rD_hi','rJ_lo','rJ_hi','gamma_lo','gamma_hi','structural_zero']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for a in rows:w.writerow(dict(id=a['id'],T=a['T'],H=a['H'],D=a['D'],rH_lo=a['r_H']['lo'],rH_hi=a['r_H']['hi'],rD_lo=a['r_D']['lo'],rD_hi=a['r_D']['hi'],rJ_lo=a['r_joint']['lo'],rJ_hi=a['r_joint']['hi'],gamma_lo=a['gamma'][0],gamma_hi=a['gamma'][1],structural_zero=a['structural_zero']))
    annual=[a for a in rows if a['T']==8760];ac=[a for a in catrows if a['id'].startswith('annual')]
    print(json.dumps({'rows':len(rows),'annual':len(annual),'positive_gamma_lower':sum(a['gamma'][0]>1e-4 for a in annual),'gamma_upper_range':[min(a['gamma'][1] for a in annual),max(a['gamma'][1] for a in annual)],'annual_catalogue_same':sum(a['status']=='same_certified' for a in ac),'annual_catalogue_total':len(ac),'verified':verification['passed'],'witnesses':len(checked),'max_residual':maximum,'elapsed_s':raw['elapsed_s']},indent=2))

if __name__=='__main__':main()
