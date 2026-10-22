"""Independent physical replay of stored annual certificates (no optimizer rerun)."""
import argparse,json
from pathlib import Path
import independent_check as independent

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--witnesses',type=Path,default=HERE/'witnesses')
parser.add_argument('--results',type=Path,default=HERE/'results.json')
parser.add_argument('--output',type=Path,default=HERE/'replay_validation.json')
a=parser.parse_args()
if not a.witnesses.is_dir():raise SystemExit('Witnesses not found: run full rye or extract the optional original archive.')
independent.HERE=a.witnesses.resolve().parent
replay=independent.replay_directory(a.witnesses)
checks=independent.verify_result_rows(json.loads(a.results.read_text()))
out={'witness_count':len(replay),'row_count':len(checks),
     'witness_failures':sum(not r['pass'] for r in replay),'row_failures':sum(not r['pass'] for r in checks),
     'replay':replay,'rows':checks}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k not in ('replay','rows')})
if not replay or not checks or out['witness_failures'] or out['row_failures']:raise SystemExit(1)
