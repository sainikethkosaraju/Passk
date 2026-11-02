"""Paired comparison of two runs on per-task pass^k, with honest inference.

Verdict rules (all must hold): minimum trials per task met on both sides; the two-stage paired
bootstrap CI excludes zero; the exact/MC paired sign-flip test rejects at alpha. Otherwise the
verdict is `inconclusive`, and the minimum detectable effect says why."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from passk import stats
from passk.logs import TrialTable

MIN_TRIALS = 5


@dataclass
class Comparison:
    k: int
    tasks: int
    delta: float  # B - A mean paired difference in pass^k
    lo: float
    hi: float
    p_value: float
    effect_size: float  # Cohen's d_z on per-task differences (nan if zero variance)
    wins: int
    losses: int
    ties: int
    mde: float  # smallest mean difference this design could detect at 80% power
    verdict: str  # regressed | improved | inconclusive
    reason: str
    p_adjusted: float | None = None  # Holm-adjusted when several k are tested
    model: str | None = None
    extra: dict = field(default_factory=dict)


def _paired(a: TrialTable, b: TrialTable, model_a: str | None, model_b: str | None, k: int):
    ca, cb = a.counts(model_a), b.counts(model_b)
    shared = sorted(set(ca) & set(cb))
    return [(t, ca[t], cb[t]) for t in shared if ca[t][0] >= k and cb[t][0] >= k]


def compare(a: TrialTable, b: TrialTable, k: int = 1, min_trials: int = MIN_TRIALS, seed: int = 0,
            alpha: float = 0.05, reps: int = 1000, model_a: str | None = None,
            model_b: str | None = None) -> Comparison:
    rows = _paired(a, b, model_a, model_b, k)
    if not rows:
        nan = float("nan")
        return Comparison(k, 0, nan, nan, nan, 1.0, nan, 0, 0, 0, nan, "inconclusive",
                          "no shared tasks with enough trials")
    short = any(min(ra[0], rb[0]) < min_trials for _, ra, rb in rows)
    diffs = [stats.ph(rb[0], rb[1], k) - stats.ph(ra[0], ra[1], k) for _, ra, rb in rows]
    d = stats.mean(diffs)
    lo, hi = stats.paired_cluster_bootstrap([r[1] for r in rows], [r[2] for r in rows], k,
                                            reps=reps, alpha=alpha, seed=seed)
    p = stats.sign_flip_test(diffs, seed=seed)
    sdv = stats.sd(diffs)
    es = d / sdv if sdv > 0 else float("nan")
    wins, losses = sum(x > 1e-12 for x in diffs), sum(x < -1e-12 for x in diffs)
    m = stats.mde(len(diffs), sdv, alpha) if sdv > 0 else 0.0
    common: dict[str, Any] = dict(k=k, tasks=len(diffs), delta=d, lo=lo, hi=hi, p_value=p, effect_size=es,
                  wins=wins, losses=losses, ties=len(diffs) - wins - losses, mde=m)
    if short:
        return Comparison(**common, verdict="inconclusive",
                          reason=f"fewer than {min_trials} trials per task on at least one side; "
                                 "refusing to call significance")
    if hi < 0 and p < alpha:
        v, r = "regressed", "paired interval below zero and sign-flip test significant"
    elif lo > 0 and p < alpha:
        v, r = "improved", "paired interval above zero and sign-flip test significant"
    else:
        v = "inconclusive"
        r = (f"no significant difference (interval [{lo:+.3f}, {hi:+.3f}], p={p:.3f}); "
             f"this design can only detect |delta| >= {m:.3f}")
    return Comparison(**common, verdict=v, reason=r)


def compare_ks(a: TrialTable, b: TrialTable, ks: list[int], **kw) -> list[Comparison]:
    """Compare at several k and Holm-adjust the p-values (family-wise error control)."""
    out = [compare(a, b, k, **kw) for k in ks]
    adj = stats.holm([c.p_value for c in out])
    for c, p in zip(out, adj):
        c.p_adjusted = p
        if c.verdict != "inconclusive" and p >= kw.get("alpha", 0.05):
            c.verdict, c.reason = "inconclusive", f"not significant after Holm correction (p_adj={p:.3f})"
    return out
