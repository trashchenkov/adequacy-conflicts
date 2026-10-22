
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Independent rational support enumeration for the H/D counterexamples.

No imports from either optimization implementation.  For the two examples,
there is exactly one charging interval for 0 <= r <= 1/2, eta=tau=1,
and charge/energy capacity is 2.  A prescribed set of zero-shortage
deficit intervals is feasible iff its total residual demand is <= 2
and each residual demand is <= the discharge limit.  Partial discharge
outside this set cannot improve either duration criterion.
"""

from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path


def duration(bits, cyclic):
    if not any(bits):
        return 0
    if all(bits):
        return len(bits)
    scan = bits + bits if cyclic else bits
    longest = run = 0
    for bit in scan:
        run = run + 1 if bit else 0
        longest = max(longest, run)
    return longest


def encode(obj):
    if isinstance(obj, F):
        return str(obj)
    if isinstance(obj, dict):
        return {key: encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(value) for value in obj]
    return obj


def validate_schedule(x, r, charge, discharge, initial, cyclic, H, D):
    x, r = list(map(F, x)), F(r)
    charge, discharge = list(map(F, charge)), list(map(F, discharge))
    states, shortage = [F(initial)], []
    for net, c, d in zip(x, charge, discharge):
        assert 0 <= c <= min(F(2), max(net + r, F(0)))
        assert 0 <= d <= min(F(2), max(-net - r, F(0)))
        assert c * d == 0
        states.append(states[-1] + c - d)
        assert 0 <= states[-1] <= 2
        shortage.append(max(-net - r, F(0)) - d)
    assert states[-1] == states[0]
    support = [ell > 0 for ell in shortage]
    h, d = sum(support), duration(support, cyclic)
    return dict(r=r, charge=charge, discharge=discharge, soc=states,
                shortage=shortage, actual_H=h, actual_D=d,
                satisfies_H=h <= H, satisfies_D=d <= D)


def check_example(name, x, H, D, cyclic, schedules):
    deficits = [t for t, value in enumerate(x) if value < 0]
    demands = {t: F(-x[t]) for t in deficits}
    rows = []
    for choose in product((False, True), repeat=len(deficits)):
        covered = [t for t, yes in zip(deficits, choose) if yes]
        shortage_support = [t in deficits and t not in covered for t in range(len(x))]
        h, d = sum(shortage_support), duration(shortage_support, cyclic)
        cost = sum((demands[t] for t in covered), F(0))
        # Exact threshold until the first deficit disappears.  All minima
        # below are in [0, 1/2], strictly below that first breakpoint (1).
        threshold = max(F(0), (cost - 2) / len(covered)) if covered else F(0)
        threshold = max([threshold] + [demands[t] - 2 for t in covered])
        rows.append(dict(covered=[t + 1 for t in covered], H=h, D=d,
                         cost_at_zero=cost, threshold=threshold,
                         H_ok=h <= H, D_ok=d <= D))
    r_H = min(row['threshold'] for row in rows if row['H_ok'])
    r_D = min(row['threshold'] for row in rows if row['D_ok'])
    r_joint = min(row['threshold'] for row in rows if row['H_ok'] and row['D_ok'])
    assert (r_H, r_D, r_joint) == (0, 0, F(1, 2))
    assert r_joint < min(demands.values())
    assert min(row['cost_at_zero'] for row in rows if row['H_ok'] and row['D_ok']) == 3
    witnesses = {
        label: validate_schedule(x, r, c, d, e0, cyclic, H, D)
        for label, r, c, d, e0 in schedules
    }
    assert witnesses['H_only']['satisfies_H'] and not witnesses['H_only']['satisfies_D']
    assert witnesses['D_only']['satisfies_D'] and not witnesses['D_only']['satisfies_H']
    assert witnesses['joint']['satisfies_H'] and witnesses['joint']['satisfies_D']
    return dict(name=name, x=x, E=2, Pc=2, Pd=2, eta=1, delta=0, tau=1,
                cyclic_SOC=True, cyclic_duration=cyclic, H=H, D=D,
                r_H=r_H, r_D=r_D, r_joint=r_joint,
                gamma_over_Pd=r_joint / 2, supports=rows, witnesses=witnesses)


def check_combinatorial_minimality():
    checked = 0
    for cyclic in (False, True):
        for T in range(1, 13):
            observed = [0] * (T + 1)
            for bits in product((False, True), repeat=T):
                actual = duration(list(bits), cyclic)
                for D in range(actual, T + 1):
                    observed[D] = max(observed[D], sum(bits))
                checked += 1
            for D in range(T + 1):
                expected = (T if D >= T else
                            T - ((T + D) // (D + 1) if cyclic else T // (D + 1)))
                assert observed[D] == expected, (cyclic, T, D, observed, expected)
                if T < (6 if cyclic else 5):
                    for H in range(T + 1):
                        assert H <= D or expected <= H
    return dict(binary_profiles_checked=checked, formula_checked_through_T=12,
                smallest_possible_T_linear=5, smallest_possible_T_cyclic=6,
                scope='All feasible profile families, not only one storage')


def main():
    linear = check_example(
        'minimal_linear', [-1, 2, -1, -2, -2], 2, 1, False,
        [('H_only', 0, [0, 2, 0, 0, 0], [1, 0, 1, 0, 0], 1),
         ('D_only', 0, [0, 2, 0, 0, 0], [0, 0, 0, 2, 0], 0),
         ('joint', F(1, 2), [0, 2, 0, 0, 0], [F(1, 2), 0, 0, F(3, 2), 0], F(1, 2))])
    cyclic = check_example(
        'minimal_cyclic', [2, -1, -1, -2, -2, -2], 3, 2, True,
        [('H_only', 0, [2, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], 0),
         ('D_only', 0, [2, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0], 0),
         ('joint', F(1, 2), [2, 0, 0, 0, 0, 0], [0, F(1, 2), 0, F(3, 2), 0, 0], 0)])
    result = dict(arithmetic='fractions.Fraction; no numerical tolerance',
                  method='Independent enumeration of fully covered interval sets and exact schedule balances',
                  examples=[linear, cyclic], minimality=check_combinatorial_minimality())
    destination = Path(__file__).with_suffix('.json')
    destination.write_text(json.dumps(encode(result), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(encode(dict(
        examples=[{key: row[key] for key in ['name', 'r_H', 'r_D', 'r_joint', 'gamma_over_Pd']}
                  for row in result['examples']], minimality=result['minimality'])),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
