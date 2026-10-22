"""Merge stage evidence and audit every sign structure after the negative correction."""
import json,collections
from pathlib import Path
from search import ROOT,structures

initial=json.loads((ROOT/'summary.json').read_text());refined=json.loads((ROOT/'refinement_summary.json').read_text())
supp=json.loads((ROOT/'negative_supplement.json').read_text())
initial_rows={(r['T'],r['negative_mask']):r for r in map(json.loads,(ROOT/'results.jsonl').read_text().splitlines())}
refinement_rows={(r['T'],r['negative_mask']):r for r in map(json.loads,(ROOT/'refinement_results.jsonl').read_text().splitlines())}
supp_rows={(r['T'],r['negative_mask']):r for r in supp['rows']}
coverage=structures();optimization_status=collections.Counter();definition_status=collections.Counter();npz_rows=0
for row in coverage:
    key=(row['T'],row['negative_mask'])
    if row['status']=='search':
        if key in initial_rows:
            row['status']='completed_initial_and_refinement';row['evaluations']=initial_rows[key]['evaluations']
            ref=refinement_rows[key]
            for a in ref['definitions']:
                definition_status[a['status']]+=1
                if a['status']=='completed':
                    optimization_status[str(a['optimizer_status'])]+=1;row['evaluations']+=a['evaluations'];npz_rows+=a['evaluations']
                assert a['status'] in ('completed','comparable')
        else:
            sup=supp_rows[key];assert sup['status']=='completed';row['status']='completed_negative_supplement'
            row['evaluations']=sum(d.get('evaluations',0) for d in sup['definitions'])
    if row['status']=='reflection':
        assert (row['T'],row['reflected_mask']) in initial_rows
assert len(coverage)==2046 and npz_rows==refined['evaluations']
counts=collections.Counter(r['status'] for r in coverage)
summary=dict(all_sign_structures=2046,coverage=dict(counts),all_accounted_for=True,all_started_searches_completed=True,
    continuous_global_optimum_proven=False,evaluations=dict(initial=initial['evaluations'],refinement=refined['evaluations'],negative_supplement=supp['evaluations']),
    total_evaluations=initial['evaluations']+refined['evaluations']+supp['evaluations'],
    compute_seconds_sum=initial['elapsed_seconds']+refined['elapsed_seconds']+supp['elapsed_seconds'],
    refinement_definition_status=dict(definition_status),refinement_local_optimizer_status=dict(optimization_status),
    best_gamma=[b['selected']['gamma'] for b in refined['best']],
    numerical_capacity_absolute_bisection_width_max=2**-27,
    fixed_support_LP_validation='engine_validation.json',exact_rational_examples='rational_check.json',
    initial_error='All-negative structures were initially excluded with an incorrect zero-conflict explanation. The six noncomparable T5..10 structures were optimized separately; T1..4 are comparable. Original evidence preserved.',
    notes='Powell status1 denotes evaluation budget reached, not a solver infeasibility or global certificate.')
(ROOT/'coverage_final.json').write_text(json.dumps(coverage,indent=2));(ROOT/'summary_final.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
