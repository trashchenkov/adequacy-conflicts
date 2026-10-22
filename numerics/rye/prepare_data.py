"""Reproducible audit and declared short internal-gap interpolation; no rescaling."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,shutil
ROOT=Path(__file__).resolve().parent
src=ROOT.parents[1]/'data/rye'
a=pd.read_csv(src/'rye_generation_and_load.csv')
t=pd.to_datetime(a['index'],utc=True)
assert not t.duplicated().any()
assert (t.diff().dropna()==pd.Timedelta(hours=1)).all()
missing=np.flatnonzero(a.Wind.isna()); groups=np.split(missing,np.where(np.diff(missing)!=1)[0]+1)
ranges=[{'start':str(t[g[0]]),'end':str(t[g[-1]]),'length':len(g),'rows':g.tolist()} for g in groups if len(g)]
assert a[['Solar','Consumption']].notna().all().all()
assert all(g[0]>0 and g[-1]<len(a)-1 for g in groups)
a['Wind_imputed']=a.Wind.isna();interpolated=a.Wind.interpolate()
for g in groups:
    a.loc[g,'Wind']=interpolated.iloc[g] if len(g)<=6 else 0.0
a['x']=a.Solar+a.Wind-a.Consumption
a.to_csv(ROOT.parents[1]/'data/rye/processed.csv',index=False)
out={'T':len(a),'start':str(t.iloc[0]),'end':str(t.iloc[-1]),'missing_wind':ranges,'long_gap_policy':'zero generation, conditional assumption, not measured','consumption_kWh':float(a.Consumption.sum()),'peak_load_kW':float(a.Consumption.max()),'x_min':float(a.x.min()),'negative_values':{k:int((a[k]<0).sum()) for k in ['Consumption','Solar','Wind']},'positive_import_q05':float(np.quantile(-a.x[a.x<0],.05)),'source_sha256':hashlib.sha256((src/'rye_generation_and_load.csv').read_bytes()).hexdigest()}
(ROOT.parents[1]/'data/rye/audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out,ensure_ascii=False))
