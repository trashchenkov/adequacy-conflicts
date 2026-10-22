
# Repository-local import path (no external working directory).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
"""Independent rational checks of the one-charge lemma and sharp example."""

from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json
import random


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: encode(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(val) for val in value]
    return value


def check_budget_lemma(seed=2026102202, count=1000):
    rng = random.Random(seed)
    largest_load_fraction = F(0)
    for _ in range(count):
        k = rng.randint(2, 10)
        q = rng.randint(1, k - 1)
        overlap = rng.randint(0, q)
        shared = [F(rng.randint(1, 20), 20) for _ in range(overlap)]
        a = sorted(F(rng.randint(1, 20), 20) for _ in range(k - overlap))
        b = [F(rng.randint(1, 20), 20) for _ in range(q - overlap)]
        budget = sum(shared) + max(sum(a), sum(b))
        chosen = shared + b + a[:k - q]
        shifted_cost = sum(max(F(0), val - F(1, 4)) for val in chosen)
        assert len(chosen) == k
        assert shifted_cost <= budget
        largest_load_fraction = max(largest_load_fraction, shifted_cost / budget)
    # Exact equality cases, not merely sampled near equality.
    equality = []
    for q in range(1, 6):
        a, b = [F(1, 2)] * (2 * q), [F(1)] * q
        cost = sum(max(F(0), val - F(1, 4)) for val in b + a[:q])
        assert cost == q
        equality.append(dict(k=2 * q, q=q, budget=q, shifted_joint_cost=cost))
    return dict(seed=seed, random_cases=count, all_passed=True,
                largest_sampled_cost_over_budget=largest_load_fraction,
                exact_equality_cases=equality)


def duration(bits, cyclic):
    if all(bits):
        return len(bits)
    longest = run = 0
    for bit in bits + bits if cyclic else bits:
        run = run + 1 if bit else 0
        longest = max(longest, run)
    return longest


def exact_example():
    demands = [F(1), F(1), F(2), F(2), F(2)]
    results = []
    for cyclic in (False, True):
        feasible_supports = []
        for covered_count in range(6):
            for covered in combinations(range(5), covered_count):
                support = [False] + [t not in covered for t in range(5)]
                if sum(support) > 3 or duration(support, cyclic) > 2:
                    continue
                cost = sum(demands[t] for t in covered)
                threshold = max(F(0), (cost - 2) / covered_count)
                feasible_supports.append(dict(covered=[t + 2 for t in covered],
                                              cost_at_zero=cost, threshold=threshold))
        exact_minimum = min(row['threshold'] for row in feasible_supports)
        assert exact_minimum == F(1, 2)
        results.append(dict(cyclic_duration=cyclic, r_H=0, r_D=0,
                            r_joint=exact_minimum, Pd=2,
                            gamma_over_Pd=exact_minimum / 2,
                            one_charge_regime_valid_until_r=1,
                            candidate_supports=feasible_supports))
    return results


def fleet_one_charge_probe(seed=2026102203, count=1000):
    """Independent finite controls of the fleet theorem proved in THEORY_RU.

    Static one-charge fleet coverage is a bipartite flow feasibility test:
    every m selected hours require sum of the m largest demands <=
    sum_i min(E_i, m*Pd_i).  The subset inequality is exact in this model.
    Shared intervals retain their old cost.  Other chosen intervals receive
    extra firm power 1/4, exactly as in the existence proof.
    """
    rng = random.Random(seed)
    failures = []
    for case in range(count):
        parts = [rng.randint(1, 10) for _ in range(rng.randint(2, 3))]
        pd = [F(value, sum(parts)) for value in parts]
        energy = [power * F(rng.randint(1, 12), 2) for power in pd]
        source = F(rng.randint(1, 12), 2)

        def cap(n):
            return min(source, sum(min(e, n * p) for e, p in zip(energy, pd)))

        def feasible(demands):
            ordered = sorted(demands, reverse=True)
            return all(sum(ordered[:m]) <= cap(m) for m in range(1, len(ordered) + 1))

        def feasible_scale(values):
            ordered = sorted(values, reverse=True)
            return min([F(1)] + [cap(m) / sum(ordered[:m]) for m in range(1, len(values) + 1)])

        k = rng.randint(2, 8)
        q = rng.randint(1, k - 1)
        overlap = rng.randint(0, q)
        shared = [F(rng.randint(1, 20), 20) for _ in range(overlap)]
        a = [F(rng.randint(1, 20), 20) for _ in range(k - overlap)]
        b = [F(rng.randint(1, 20), 20) for _ in range(q - overlap)]
        factor = min(feasible_scale(shared + a), feasible_scale(shared + b))
        shared, a, b = [[value * factor for value in items] for items in (shared, a, b)]
        assert feasible(shared + a) and feasible(shared + b)
        chosen = b + sorted(a)[:k - q]
        reduced = [max(F(0), value - F(1, 4)) for value in chosen]
        if not feasible(shared + reduced):
            failures.append(dict(case=case, k=k, q=q, a=a, b=b,
                                 shared=shared, Pd=pd, E=energy, source=source,
                                 reduced_joint=reduced))
            # Report a failed extension promptly; do not optimize it silently.
            break
    return dict(seed=seed, cases_attempted=case + 1, proposed_extension_failures=failures,
                status='rational controls of the proved one-charge fleet theorem; not a global search')


def main():
    result = dict(arithmetic='fractions.Fraction',
                  scalar_lemma=check_budget_lemma(),
                  sharp_examples=exact_example(),
                  fleet_one_charge_probe=fleet_one_charge_probe())
    Path(__file__).with_suffix('.json').write_text(
        json.dumps(encode(result), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(encode(dict(
        scalar_lemma=result['scalar_lemma'],
        exact_minima=[row['r_joint'] for row in result['sharp_examples']],
        fleet_one_charge_probe=result['fleet_one_charge_probe'])), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
