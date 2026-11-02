"""Public estimator API. Thin, validated wrappers over `passk.stats`."""

from __future__ import annotations

import math
from collections.abc import Sequence

from passk import stats
from passk.stats import mean, wilson  # noqa: F401  (re-exported)


def _check(n: int, c: int, k: int) -> None:
    if not (0 <= c <= n):
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")


def pass_hat_k(n: int, c: int, k: int) -> float:
    """P(k trials drawn without replacement from n recorded trials all succeed) = C(c,k)/C(n,k)."""
    _check(n, c, k)
    return stats.ph(n, c, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k drawn trials succeeds) = 1 - C(n-c,k)/C(n,k)."""
    _check(n, c, k)
    return stats.pa(n, c, k)


def bootstrap_ci(values: Sequence[float], *, reps: int = 2000, alpha: float = 0.05, seed: int = 0,
                 method: str = "percentile") -> tuple[float, float]:
    if not values:
        raise ValueError("no values")
    return (stats.bca_ci if method == "bca" else stats.percentile_ci)(values, reps, alpha, seed)


def aggregate(counts: Sequence[tuple[int, int]], k: int, *, kind: str = "hat", seed: int = 0,
              method: str = "cluster", reps: int = 1000) -> dict[str, float]:
    """Mean per-task metric over tasks with n >= k.

    method="cluster": two-stage bootstrap (tasks, then trials within tasks) - the default.
    method="tasks": task-only percentile bootstrap (ignores within-task trial noise; for comparison)."""
    fn = pass_hat_k if kind == "hat" else pass_at_k
    kept = [(n, c) for n, c in counts if n >= k]
    excluded = len(counts) - len(kept)
    if not kept:
        nan = float("nan")
        return {"value": nan, "lo": nan, "hi": nan, "tasks": 0, "excluded": excluded}
    vals = [fn(n, c, k) for n, c in kept]
    if method == "cluster":
        lo, hi = stats.cluster_bootstrap_pass(kept, k, kind, reps=reps, seed=seed)
    else:
        lo, hi = stats.percentile_ci(vals, reps=reps, seed=seed)
    v = mean(vals)
    return {"value": v, "lo": min(lo, v), "hi": max(hi, v), "tasks": len(kept), "excluded": excluded}


__all__ = ["aggregate", "bootstrap_ci", "mean", "pass_at_k", "pass_hat_k", "wilson", "math"]
