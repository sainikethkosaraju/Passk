"""Statistical inference for reliability metrics. Pure standard library.

Unit of analysis: the TASK. Each task t has n_t recorded trials with c_t successes. The
per-task reliability estimand is p_t^k (all k independent future trials succeed); the headline
number is its mean over tasks. Uncertainty has two sources, which the two-stage ("cluster")
bootstrap captures and a task-only bootstrap does not:
  1. which tasks were sampled (between-task variance),
  2. how few trials we saw per task (within-task variance).
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from functools import cache
from statistics import NormalDist

_N = NormalDist()


# ---------- estimators ------------------------------------------------------------------------
@cache
def ph(n: int, c: int, k: int) -> float:
    """Unbiased pass^k estimator C(c,k)/C(n,k) (same estimator as Inspect's `pass_k` reducer)."""
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


@cache
def pa(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator 1 - C(n-c,k)/C(n,k) (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------- Beta distribution (incomplete beta by continued fraction) ---------------------------
def _betacf(a: float, b: float, x: float) -> float:
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = d if abs(d) > tiny else tiny
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = d if abs(d) > tiny else tiny
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln_front) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def jeffreys(c: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    a, b = c + 0.5, n - c + 0.5
    lo = 0.0 if c == 0 else beta_ppf(alpha / 2, a, b)
    hi = 1.0 if c == n else beta_ppf(1 - alpha / 2, a, b)
    return lo, hi


def posterior_pass_hat(c: int, n: int, k: int, prior: tuple[float, float] = (0.5, 0.5),
                       alpha: float = 0.05) -> dict[str, float]:
    """Bayesian reliability for ONE task: p ~ Beta(a+c, b+n-c); target p^k.

    Unlike C(c,k)/C(n,k) this is defined for k > n (extrapolating e.g. pass^10 from 5 trials) and
    shrinks extreme counts. Interval = monotone transform of the Beta quantiles of p."""
    a, b = prior[0] + c, prior[1] + n - c
    mean = math.exp(math.lgamma(a + k) + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(a + b + k))
    return {"mean": mean, "lo": beta_ppf(alpha / 2, a, b) ** k, "hi": beta_ppf(1 - alpha / 2, a, b) ** k}


# ---------- bootstrap -------------------------------------------------------------------------
def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    pos = q * (len(sorted_xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def percentile_ci(values: Sequence[float], reps: int = 2000, alpha: float = 0.05, seed: int = 0,
                  stat: Callable[[Sequence[float]], float] | None = None) -> tuple[float, float]:
    stat = stat or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    n = len(values)
    s = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    return _quantile(s, alpha / 2), _quantile(s, 1 - alpha / 2)


def bca_ci(values: Sequence[float], reps: int = 2000, alpha: float = 0.05, seed: int = 0,
           stat: Callable[[Sequence[float]], float] | None = None) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval (Efron 1987) for a statistic of iid values."""
    stat = stat or (lambda xs: sum(xs) / len(xs))
    n = len(values)
    if n < 3 or len(set(values)) == 1:
        return percentile_ci(values, reps, alpha, seed, stat)
    rng = random.Random(seed)
    theta = stat(values)
    boots = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    frac = (sum(b < theta for b in boots) + 0.5 * sum(b == theta for b in boots)) / reps
    frac = min(max(frac, 1 / (2 * reps)), 1 - 1 / (2 * reps))
    z0 = _N.inv_cdf(frac)
    jack = [stat(list(values[:i]) + list(values[i + 1:])) for i in range(n)]
    jm = sum(jack) / n
    num = sum((jm - j) ** 3 for j in jack)
    den = 6.0 * (sum((jm - j) ** 2 for j in jack) ** 1.5)
    acc = num / den if den else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = _N.inv_cdf(q)
        adj = _N.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        out.append(_quantile(boots, min(max(adj, 0.0), 1.0)))
    return out[0], out[1]


def _draw_p(rng: random.Random, n: int, c: int, smooth: bool) -> float:
    """Within-task success probability for the inner resample. Plug-in c/n gives ZERO variance for
    tasks at 0/n or n/n (measured: pass^5 coverage 81%, docs/coverage.md); the Jeffreys posterior
    Beta(c+1/2, n-c+1/2) restores honest within-task uncertainty."""
    return rng.betavariate(c + 0.5, n - c + 0.5) if smooth else c / n


def cluster_bootstrap_pass(tasks: Sequence[tuple[int, int]], k: int, kind: str = "hat",
                           reps: int = 2000, alpha: float = 0.05, seed: int = 0,
                           smooth: bool = True) -> tuple[float, float]:
    """Two-stage bootstrap for mean_t estimator(n_t, c_t, k): resample tasks, then resample each
    drawn task's trials. Tasks with n_t < k are excluded by the caller."""
    est = ph if kind == "hat" else pa
    rng = random.Random(seed)
    m = len(tasks)
    vals = []
    for _ in range(reps):
        tot = 0.0
        for _ in range(m):
            n, c = tasks[rng.randrange(m)]
            p = _draw_p(rng, n, c, smooth)
            c2 = sum(rng.random() < p for _ in range(n))
            tot += est(n, c2, k)
        vals.append(tot / m)
    vals.sort()
    return _quantile(vals, alpha / 2), _quantile(vals, 1 - alpha / 2)


def paired_cluster_bootstrap(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]], k: int,
                             reps: int = 2000, alpha: float = 0.05, seed: int = 0,
                             smooth: bool = True) -> tuple[float, float]:
    """CI for mean_t [est(b_t) - est(a_t)] with tasks paired, trials resampled independently per side."""
    rng = random.Random(seed)
    m = len(a)
    vals = []
    for _ in range(reps):
        tot = 0.0
        for _ in range(m):
            i = rng.randrange(m)
            (na, ca), (nb, cb) = a[i], b[i]
            pa_, pb_ = _draw_p(rng, na, ca, smooth), _draw_p(rng, nb, cb, smooth)
            ca2 = sum(rng.random() < pa_ for _ in range(na))
            cb2 = sum(rng.random() < pb_ for _ in range(nb))
            tot += ph(nb, cb2, k) - ph(na, ca2, k)
        vals.append(tot / m)
    vals.sort()
    return _quantile(vals, alpha / 2), _quantile(vals, 1 - alpha / 2)


# ---------- hypothesis tests --------------------------------------------------------------------
def sign_flip_test(diffs: Sequence[float], reps: int = 20000, seed: int = 0, exact_max: int = 16,
                   ) -> float:
    """Two-sided paired randomisation test of mean(diffs) == 0 (exact for <= exact_max nonzero pairs)."""
    d = [x for x in diffs if x != 0.0]
    if not d:
        return 1.0
    obs = abs(sum(d))
    n = len(d)
    eps = 1e-12
    if n <= exact_max:
        hits = sum(abs(sum(s * x for s, x in zip(signs, d))) >= obs - eps
                   for signs in itertools.product((1, -1), repeat=n))
        return hits / 2 ** n
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - eps for _ in range(reps))
    return (hits + 1) / (reps + 1)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (controls family-wise error)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


# ---------- power ---------------------------------------------------------------------------------
def mde(n_tasks: int, sd_diff: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean paired difference (normal approximation)."""
    if n_tasks < 2:
        return float("inf")
    return (_N.inv_cdf(1 - alpha / 2) + _N.inv_cdf(power)) * sd_diff / math.sqrt(n_tasks)


def required_tasks(delta: float, sd_diff: float, alpha: float = 0.05, power: float = 0.8) -> int:
    if delta <= 0:
        raise ValueError("delta must be positive")
    z = _N.inv_cdf(1 - alpha / 2) + _N.inv_cdf(power)
    return math.ceil((z * sd_diff / delta) ** 2)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
