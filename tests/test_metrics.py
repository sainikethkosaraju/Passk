import itertools
import math
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from passk.metrics import aggregate, bootstrap_ci, pass_at_k, pass_hat_k, wilson


def enum_hat(n, c, k):
    trials = [1] * c + [0] * (n - c)
    combos = list(itertools.combinations(range(n), k))
    return sum(all(trials[i] for i in cmb) for cmb in combos) / len(combos)


def enum_at(n, c, k):
    trials = [1] * c + [0] * (n - c)
    combos = list(itertools.combinations(range(n), k))
    return sum(any(trials[i] for i in cmb) for cmb in combos) / len(combos)


@pytest.mark.parametrize("n", range(1, 8))
def test_matches_exact_enumeration(n):
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_hat_k(n, c, k) == pytest.approx(enum_hat(n, c, k))
            assert pass_at_k(n, c, k) == pytest.approx(enum_at(n, c, k))


@given(st.integers(1, 30).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n))))
def test_invariants(nc):
    n, c = nc
    assert pass_hat_k(n, c, 1) == pytest.approx(pass_at_k(n, c, 1)) == pytest.approx(c / n)
    hats = [pass_hat_k(n, c, k) for k in range(1, n + 1)]
    ats = [pass_at_k(n, c, k) for k in range(1, n + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(hats, hats[1:]))
    assert all(a <= b + 1e-12 for a, b in zip(ats, ats[1:]))
    assert all(h <= a + 1e-12 for h, a in zip(hats, ats))


def test_bad_inputs():
    with pytest.raises(ValueError):
        pass_hat_k(3, 4, 1)
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 4)


def test_wilson_basic():
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0


def test_wilson_coverage():
    rng = random.Random(7)
    n, p, reps, hit = 20, 0.7, 2000, 0
    for _ in range(reps):
        c = sum(rng.random() < p for _ in range(n))
        lo, hi = wilson(c, n)
        hit += lo <= p <= hi
    assert 0.92 <= hit / reps <= 0.99  # nominal 0.95


def test_bootstrap_coverage_over_tasks():
    rng = random.Random(11)
    true_mean = 0.6
    reps, hit = 300, 0
    for r in range(reps):
        vals = [float(rng.random() < true_mean) for _ in range(40)]
        lo, hi = bootstrap_ci(vals, reps=300, seed=r)
        hit += lo <= true_mean <= hi
    assert 0.90 <= hit / reps <= 0.99


def test_aggregate_excludes_short_tasks():
    agg = aggregate([(5, 5), (2, 2)], k=3)
    assert agg["tasks"] == 1 and agg["excluded"] == 1 and agg["value"] == 1.0
    assert math.isnan(aggregate([(2, 2)], k=3)["value"])
