"""Rebuild the Rye summary from frozen results; this is NOT an annual MILP rerun."""
from pathlib import Path
import csv,json,math,sys
ROOT=Path(__file__).resolve().parents[1]
ROWS=[]
def check(name,expected,obtained,tol=1e-8):
    ok=math.isfinite(obtained) and abs(obtained-expected)<=tol
    ROWS.append(dict(name=name,expected=expected,obtained=obtained,tolerance=tol,ok=ok))
    print(f'{"ok" if ok else "FAIL"} {name}: expected={expected:.12g}, obtained={obtained:.12g}',flush=True)

saved=json.loads((ROOT/'numerics/rye/results.json').read_text())
table=list(csv.DictReader((ROOT/'numerics/rye/summary.csv').open()))
index={r['id']:r for r in table}
recomputed=[];matches=0
for r in saved['rows']:
    a,b,j=r['r1'],r['r2'],r['rjoint']
    low=max(0,j['lo']-max(a['hi'],b['hi']))
    high=max(0,j['hi']-max(a['lo'],b['lo']))
    old=index[r['id']]
    values={'gamma_lo_kW':low,'gamma_hi_kW':high,
            'r1_lo_kW':a['lo'],'r1_hi_kW':a['hi'],'r2_lo_kW':b['lo'],'r2_hi_kW':b['hi'],
            'rjoint_lo_kW':j['lo'],'rjoint_hi_kW':j['hi']}
    matches+=int(all(abs(float(old[k])-v)<1e-8 for k,v in values.items()))
    recomputed.append(dict(id=r['id'],**values))
check('Rye table rows reconstructed',len(table),matches,0)
with (ROOT/'validation/rye_table_recomputed.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(recomputed[0]));writer.writeheader();writer.writerows(recomputed)
hd=[r for r in saved['rows'] if r['family']=='HD']
check('Rye HD pairs',24,len(hd),0)
check('Rye HD positive lower bounds',0,sum(r['gamma'][0]>1e-6 for r in hd),0)
check('Rye HD max upper kW',.087,max(r['gamma'][1] for r in hd),.0005)
check('Rye HD max upper percent',.22,100*max(r['gamma_over_sumPd'][1] for r in hd),.005)
r=next(r for r in saved['rows'] if r['id']=='tank0.02_H150E0.01')
check('Rye tank lower kW',.014,r['gamma'][0],.0005)
check('Rye tank upper kW',.048,r['gamma'][1],.0005)
check('Rye tank lower / deltaE',.012,r['gamma_over_deltaE'][0],.0005)
check('Rye tank upper / deltaE',.040,r['gamma_over_deltaE'][1],.0005)
audit=json.loads((ROOT/'data/rye/audit.json').read_text())
check('Rye hours',8280,audit['T'],0)
check('Rye peak load kW',70,audit['peak_load_kW'],.5)
# Preserve the discrepancy rather than falsely marking the manuscript's 3--5 h claim ok.
series=[r['r1'].get('D_linear',0) for r in hd]
notes={'manuscript_HD_H_witness_series_h':[3,5],'saved_range_h':[min(series),max(series)],
       'status':'DISCREPANCY' if max(series)>5 else 'ok'}
print(notes['status'], 'Rye H-only series: manuscript 3–5 h; saved',notes['saved_range_h'])
(ROOT/'validation/quick_rye.json').write_text(json.dumps({'checks':ROWS,'discrepancy':notes},indent=2)+'\n')
if not all(r['ok'] for r in ROWS):raise SystemExit(1)
