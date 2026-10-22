"""Execute isolated, timed reproduction blocks. A nonzero/timeout never means pass."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
# Each block is individually limited to 30 minutes. Search refinement depends on search.
BLOCKS = {
    'theory': [('checks/quick.py',), ('checks/exact_theory.py',), ('checks/g3/g3_analytic.py',), ('checks/g5/g5_t3.py',), ('checks/g5/g5_coast.py',), ('checks/g5/g5_coast_rand.py',)],
    'monotone': [('checks/monotone/run_checks.py',), ('checks/monotone/secondary_checks.py',)],
    'duration': [('checks/g4/g4_check.py',), ('checks/g4/g4_adv.py',)],
    'abundant': [('checks/g6/g6_abstract.py',), ('checks/g6/g6_verify.py',)],
    'multicharge': [('checks/g6/g6_search.py',), ('checks/g6/g6_more.py',), ('checks/g6/g6_fleet2.py',), ('checks/g6/g6_fleet3.py',)],
    'selfdischarge': [('checks/g2/g2_analysis.py',)],
    'figures': [('figures/make_figures.py',)],
    'rye': [('numerics/rye/prepare_data.py',), ('numerics/rye/test_model.py',), ('numerics/rye/independent_check.py', '--controls'), ('numerics/rye/run.py',), ('numerics/rye/replay.py',)],
    'yakutia': [('numerics/yakutia/check_small.py',), ('numerics/yakutia/run.py',), ('numerics/yakutia/summarize.py',)],
    'search': [('numerics/search/validate_engine.py',), ('numerics/search/search.py', '--seconds','1100','--evals','650','--historical-initial')],
    'refinement': [('numerics/search/refine.py',), ('numerics/search/negative_supplement.py',), ('numerics/search/rational_check.py',), ('numerics/search/summarize.py',)],
}

def full(blocks, work):
    work.mkdir(parents=True, exist_ok=True)
    for block in blocks:
        target=work/('search' if block=='refinement' else block)
        if block=='refinement':
            if not (target/'numerics/search/results.jsonl').exists():
                raise RuntimeError('Run full search with this --work directory before refinement')
        else:
            if target.exists():
                raise RuntimeError(f'Refusing to overwrite {target}; choose another --work directory')
            target.mkdir()
            for name in ('src','checks','numerics','figures','data'):
                shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.dylib','*.so','output'))
            (target/'validation').mkdir()
            shutil.copy2(ROOT/'validation/solver_equivalence.json',target/'validation/solver_equivalence.json')
        results=[]; started=time.monotonic()
        for command in BLOCKS[block]:
            log=work/(block+'_'+Path(command[0]).stem+'.txt')
            remaining=1800-(time.monotonic()-started)
            t=time.monotonic()
            try:
                with log.open('w') as f:
                    p=subprocess.run([sys.executable,*command],cwd=target,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,remaining))
                code=p.returncode; status='ok' if code==0 else 'FAIL'
            except subprocess.TimeoutExpired:
                code=None;status='TIMEOUT'
            results.append(dict(command=list(command),elapsed_s=time.monotonic()-t,returncode=code,status=status,log=log.name))
            print(block,command[0],status,round(results[-1]['elapsed_s'],2),flush=True)
            (work/(block+'.json')).write_text(json.dumps(results,indent=2)+'\n')
            if status!='ok':
                raise RuntimeError(f'{block} did not finish: see {log}')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('level',choices=('quick','full'))
    parser.add_argument('blocks',nargs='*',choices=tuple(BLOCKS))
    parser.add_argument('--work',type=Path,default=Path('runs')/time.strftime('%Y%m%d_%H%M%S'))
    a=parser.parse_args()
    os.chdir(ROOT)
    if a.level=='quick':
        for script in ('checks/quick.py','numerics/quick.py'):
            subprocess.run([sys.executable,script],check=True)
    else:
        full(a.blocks or list(BLOCKS),a.work.resolve())
