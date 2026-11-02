"""Estimators and intervals.

pass^k (all k sampled trials succeed): unbiased estimate C(c,k)/C(n,k).
pass@k (at least one of k succeeds): 1 - C(n-c,k)/C(n,k)  (Chen et al., 2021).
Both are exact for sampling k of n recorded trials without replacement.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def _check(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n):
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Probability that k trials drawn without replacement from n all succeed."""
    _check(n, c, k)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that at least one of k trials drawn from n succeeds."""
    _check(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = c / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(
    values: Sequence[float], *, reps: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling tasks (the unit of analysis)."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    lo = means[int((alpha / 2) * reps)]
    hi = means[min(reps - 1, int((1 - alpha / 2) * reps))]
    return lo, hi


def aggregate(
    counts: Sequence[tuple[int, int]], k: int, *, kind: str = "hat", seed: int = 0
) -> dict[str, float]:
    """Mean per-task metric over tasks with n >= k, with a bootstrap-over-tasks interval.

    `counts` is a list of (n, c) per task. Tasks with fewer than k trials are excluded
    and reported via `excluded`.
    """
    fn = pass_hat_k if kind == "hat" else pass_at_k
    vals = [fn(n, c, k) for n, c in counts if n >= k]
    excluded = len([1 for n, _ in counts if n < k])
    if not vals:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "tasks": 0, "excluded": excluded}
    lo, hi = bootstrap_ci(vals, seed=seed)
    return {"value": mean(vals), "lo": lo, "hi": hi, "tasks": len(vals), "excluded": excluded}
