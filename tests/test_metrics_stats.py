import itertools
import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from passk import stats
from passk.metrics import aggregate, pass_at_k, pass_hat_k


def enum(n, c, k, fn):
    t = [1] * c + [0] * (n - c)
    combos = list(itertools.combinations(range(n), k))
    return sum(fn(t[i] for i in cmb) for cmb in combos) / len(combos)


@pytest.mark.parametrize("n", range(1, 8))
def test_estimators_equal_exact_enumeration(n):
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_hat_k(n, c, k) == pytest.approx(enum(n, c, k, all))
            assert pass_at_k(n, c, k) == pytest.approx(enum(n, c, k, any))


@given(st.integers(1, 30).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n))))
def test_invariants(nc):
    n, c = nc
    assert pass_hat_k(n, c, 1) == pytest.approx(pass_at_k(n, c, 1)) == pytest.approx(c / n)
    hats = [pass_hat_k(n, c, k) for k in range(1, n + 1)]
    ats = [pass_at_k(n, c, k) for k in range(1, n + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(hats, hats[1:]))
    assert all(a <= b + 1e-12 for a, b in zip(ats, ats[1:]))
    assert all(h <= a + 1e-12 for h, a in zip(hats, ats))


@given(st.integers(2, 25), st.floats(0.05, 0.95))
@settings(max_examples=30, deadline=None)
def test_pass_hat_is_unbiased_for_p_to_the_k(n, p):
    """E[C(c,k)/C(n,k)] = p^k exactly when c ~ Binomial(n, p)."""
    for k in (1, 2, min(3, n)):
        exp = sum(math.comb(n, c) * p**c * (1 - p) ** (n - c) * pass_hat_k(n, c, k) for c in range(n + 1))
        assert exp == pytest.approx(p**k, rel=1e-9, abs=1e-12)


def test_bad_inputs():
    with pytest.raises(ValueError):
        pass_hat_k(3, 4, 1)
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 4)


# ---- special functions ---------------------------------------------------------------------
def test_betainc_known_values():
    assert stats.betainc(1, 1, 0.3) == pytest.approx(0.3)
    assert stats.betainc(2, 2, 0.5) == pytest.approx(0.5)
    assert stats.betainc(0.5, 0.5, 0.5) == pytest.approx(0.5)
    assert stats.betainc(2, 5, 0.3) == pytest.approx(0.579825, abs=1e-5)  # reference value
    assert stats.betainc(3, 7, 0.0) == 0.0 and stats.betainc(3, 7, 1.0) == 1.0


@given(st.floats(0.2, 20), st.floats(0.2, 20), st.floats(0.01, 0.99))
@settings(max_examples=40, deadline=None)
def test_beta_ppf_inverts_cdf(a, b, q):
    assert stats.betainc(a, b, stats.beta_ppf(q, a, b)) == pytest.approx(q, abs=1e-8)


def test_wilson_jeffreys_and_posterior():
    lo, hi = stats.wilson(5, 10)
    assert 0.23 < lo < 0.5 < hi < 0.77
    jl, jh = stats.jeffreys(0, 10)
    assert jl == 0.0 and 0.1 < jh < 0.25  # known: Jeffreys upper bound for 0/10 is ~0.175
    assert stats.jeffreys(10, 10)[1] == 1.0
    post = stats.posterior_pass_hat(5, 5, 3)
    # closed form E[p^3] for Beta(5.5, 0.5)
    expect = (5.5 * 6.5 * 7.5) / (6.0 * 7.0 * 8.0)
    assert post["mean"] == pytest.approx(expect)
    ext = stats.posterior_pass_hat(5, 5, 10)  # extrapolates beyond n
    assert 0 < ext["mean"] < post["mean"] and ext["lo"] <= ext["mean"] <= 1.0
    assert stats.posterior_pass_hat(0, 5, 2)["mean"] < 0.05


def test_sign_flip_exact_and_holm_and_power():
    d = [1.0] * 6
    assert stats.sign_flip_test(d) == pytest.approx(2 / 64)  # only all-plus / all-minus reach |sum|=6
    assert stats.sign_flip_test([0.0, 0.0]) == 1.0
    assert stats.sign_flip_test([1, -1, 1, -1]) == pytest.approx(1.0)
    big = stats.sign_flip_test([0.3] * 30 + [-0.01] * 3, reps=4000)
    assert big < 0.01
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert stats.holm([0.5]) == [0.5]
    assert stats.mde(40, 0.2) == pytest.approx((1.959964 + 0.841621) * 0.2 / math.sqrt(40), rel=1e-4)
    assert stats.required_tasks(0.1, 0.2) == math.ceil(((1.959964 + 0.841621) * 0.2 / 0.1) ** 2)
    with pytest.raises(ValueError):
        stats.required_tasks(0, 0.2)


# ---- simulation: intervals reach nominal coverage -------------------------------------------
def simulate_tasks(rng, m, n, a=2.0, b=1.0):
    """Task reliabilities p_t ~ Beta(a, b); n Bernoulli trials per task."""
    tasks = []
    for _ in range(m):
        p = rng.betavariate(a, b)
        tasks.append((n, sum(rng.random() < p for _ in range(n))))
    return tasks


@pytest.mark.parametrize("k", [1, 3])
def test_cluster_bootstrap_coverage_of_population_pass_hat(k):
    """Target = E_p[p^k] for p ~ Beta(2,1) = 2/(k+2). 40 tasks x 5 trials."""
    rng = random.Random(1234 + k)
    truth = 2.0 / (k + 2)
    reps, hit = 200, 0
    for r in range(reps):
        t = simulate_tasks(rng, 40, 5)
        lo, hi = stats.cluster_bootstrap_pass(t, k, reps=200, seed=r)
        hit += lo <= truth <= hi
    assert 0.9 <= hit / reps <= 1.0, hit / reps  # smoothed interval is conservative


def test_bca_and_percentile_cover_a_skewed_mean():
    rng = random.Random(5)
    truth = 0.15  # mean of Bernoulli(0.15): skewed small-sample case
    cov = {"pct": 0, "bca": 0}
    reps = 300
    for r in range(reps):
        xs = [float(rng.random() < truth) for _ in range(25)]
        cov["pct"] += (lambda ci: ci[0] <= truth <= ci[1])(stats.percentile_ci(xs, 300, seed=r))
        cov["bca"] += (lambda ci: ci[0] <= truth <= ci[1])(stats.bca_ci(xs, 300, seed=r))
    assert cov["bca"] / reps >= 0.88 and cov["pct"] / reps >= 0.85


def test_aggregate_methods_and_exclusion():
    agg = aggregate([(5, 5), (2, 2)], k=3)
    assert agg["tasks"] == 1 and agg["excluded"] == 1 and agg["value"] == 1.0
    assert math.isnan(aggregate([(2, 2)], k=3)["value"])
    t = [(5, c) for c in (5, 4, 4, 3, 0)]
    cl, tk = aggregate(t, 2, method="cluster"), aggregate(t, 2, method="tasks")
    assert cl["value"] == tk["value"]
    assert (cl["hi"] - cl["lo"]) >= (tk["hi"] - tk["lo"]) * 0.8  # cluster interval accounts for trial noise


def test_smoothing_fixes_high_k_undercoverage():
    """Regression test for the measured defect: the plug-in two-stage bootstrap undercovered pass^5
    (81%, docs/coverage.md). The Jeffreys-smoothed default must cover."""
    rng = random.Random(99)
    truth, reps, hit = 2.0 / 7, 120, 0
    for r in range(reps):
        t = simulate_tasks(rng, 40, 5)
        lo, hi = stats.cluster_bootstrap_pass(t, 5, reps=150, seed=r)
        hit += lo <= truth <= hi
    assert hit / reps >= 0.9
