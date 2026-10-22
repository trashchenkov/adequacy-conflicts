"""Bounded seeded differential evolution separately for each duration definition."""
import json,time
import numpy as np
from scipy.optimize import minimize
from search import ROOT,evaluate

def main():
    start=time.monotonic();rng=np.random.default_rng(2026102202);budget=1000.;population=40;generations=60
    rows=[json.loads(s) for s in (ROOT/'results.jsonl').read_text().splitlines()]
    archive=ROOT/'refinement_evaluations';archive.mkdir(exist_ok=True)
    config=dict(seed=2026102202,budget_seconds=budget,population=population,generations=generations,
        local_maxfev=450,structures=len(rows),definitions=2,trace_columns='x[0:T], E, Pc, linear(gamma,H,D,rH,rD,rjoint), cyclic(same)')
    (ROOT/'refinement_config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config),flush=True)
    best=json.loads((ROOT/'best.json').read_text());out=open(ROOT/'refinement_results.jsonl','w');total=done=0
    for row in rows:
        T=row['T'];neg=row['negative_mask'];sg=np.array([-1 if neg&(1<<t) else 1 for t in range(T)])
        upper=np.array([2. if s<0 else float(T) for s in sg]+[float(T),float(T)])
        records=[];result=dict(T=T,negative_mask=neg,definitions=[])
        for j in range(2):
            if time.monotonic()-start>budget:
                result['definitions'].append(dict(cyclic=bool(j),status='budget_not_started'));continue
            if not row['noncomparable_pairs'][j]:
                result['definitions'].append(dict(cyclic=bool(j),status='comparable'));continue
            count=0;localbest=row['best'][j];bestscore=localbest['selected']['gamma'];bestp=np.r_[np.abs(localbest['x']),localbest['E'],localbest['Pc']]
            def objective(p):
                nonlocal count,total,localbest,bestscore,bestp
                x=p[:T]*sg;ans=evaluate(x,float(p[T]),float(p[T+1]));count+=1;total+=1
                record=list(x)+[float(p[T]),float(p[T+1])]
                for a in ans:record.extend(a[k] for k in ('gamma','H','D','r_H','r_D','r_joint'))
                records.append(record)
                rec=dict(x=x.tolist(),E=float(p[T]),Pc=float(p[T+1]),Pd=1,eta=1,structure=[T,neg],results=ans,selected=ans[j],evaluation=total)
                if ans[j]['gamma']>bestscore+1e-9:bestscore=ans[j]['gamma'];bestp=np.array(p);localbest=rec
                for q,a in enumerate(ans):
                    if a['gamma']>best[q]['selected']['gamma']+1e-8:
                        best[q]=dict(rec,selected=a);(ROOT/'refinement_best.json').write_text(json.dumps(best,indent=2))
                        print(json.dumps(dict(event='best',cyclic=q,gamma=a['gamma'],T=T,negative_mask=neg,x=x.tolist(),E=float(p[T]),Pc=float(p[T+1]),H=a['H'],D=a['D'])),flush=True)
                return -ans[j]['gamma']
            pop=np.empty((population,T+2));pop[0]=bestp
            for z in range(1,population):
                p=np.ones(T+2);p[:T]=rng.uniform(.02,1.6,T)
                p[T:]=rng.uniform(.15,min(T,4.),2)
                if z<population//2:
                    p[:T][sg<0]=rng.choice([.25,.5,.75,1.,1.25,1.5],sum(sg<0))
                    p[:T][sg>0]=rng.choice([.25,.5,1.,2.],sum(sg>0))
                    p[T:]=rng.choice([.5,.75,1.,1.5,2.,3.],2)
                pop[z]=np.minimum(p,upper)
            values=np.array([objective(p) for p in pop])
            for generation in range(generations):
                elite=pop[np.argmin(values)].copy()
                for z in range(population):
                    a,b,c=rng.choice(population,3,replace=False)
                    # Alternate rand/1 and current-to-best/1 to keep diverse restarts.
                    F=rng.uniform(.45,.95)
                    mutant=pop[a]+F*(pop[b]-pop[c]) if generation%2 else pop[z]+F*(elite-pop[z])+F*(pop[a]-pop[b])
                    trial=np.where(rng.random(T+2)<.75,mutant,pop[z]);trial[rng.integers(T+2)]=mutant[rng.integers(T+2)]
                    trial=np.clip(trial,0.,upper)
                    score=objective(trial)
                    if score<=values[z]+1e-12:pop[z]=trial;values[z]=score
            opt=minimize(objective,bestp,method='Powell',bounds=list(zip(np.zeros(T+2),upper)),options=dict(maxfev=450,maxiter=4,xtol=1e-5,ftol=1e-7))
            result['definitions'].append(dict(cyclic=bool(j),status='completed',evaluations=count,best=localbest,
                optimizer_status=int(opt.status),optimizer_message=str(opt.message)));done+=1
        np.savez_compressed(archive/f'T{T}_mask{neg}.npz',evaluations=np.array(records))
        out.write(json.dumps(result,separators=(',',':'))+'\n');out.flush()
        if done%40<2:print(json.dumps(dict(event='progress',completed=done,evaluations=total,seconds=time.monotonic()-start)),flush=True)
    out.close();summary=dict(config=config,elapsed_seconds=time.monotonic()-start,evaluations=total,completed=done,best=best)
    (ROOT/'refinement_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
