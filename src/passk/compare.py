"""Paired version comparison on per-task pass^k."""

from __future__ import annotations

from dataclasses import dataclass

from passk.logs import TrialTable
from passk.metrics import bootstrap_ci, mean, pass_hat_k

MIN_TRIALS = 5


@dataclass
class Comparison:
    k: int
    tasks: int
    delta: float  # B - A, mean paired difference in pass^k
    lo: float
    hi: float
    effect_size: float  # Cohen's d_z on paired differences (nan if no variance)
    verdict: str  # regressed | improved | inconclusive
    reason: str


def compare(a: TrialTable, b: TrialTable, k: int = 1, min_trials: int = MIN_TRIALS,
            seed: int = 0) -> Comparison:
    ca, cb = a.counts(), b.counts()
    shared = sorted(set(ca) & set(cb))
    diffs: list[float] = []
    short = False
    for t in shared:
        (na, xa), (nb, xb) = ca[t], cb[t]
        if na < k or nb < k:
            continue
        if min(na, nb) < min_trials:
            short = True
        diffs.append(pass_hat_k(nb, xb, k) - pass_hat_k(na, xa, k))
    if not diffs:
        return Comparison(k, 0, float("nan"), float("nan"), float("nan"), float("nan"),
                          "inconclusive", "no shared tasks with enough trials")
    d = mean(diffs)
    lo, hi = bootstrap_ci(diffs, seed=seed)
    var = sum((x - d) ** 2 for x in diffs) / max(1, len(diffs) - 1)
    sd = var ** 0.5
    es = d / sd if sd > 0 else float("nan")
    if short:
        return Comparison(k, len(diffs), d, lo, hi, es, "inconclusive",
                          f"fewer than {min_trials} trials per task on at least one side; "
                          "refusing to call significance")
    if hi < 0:
        v, r = "regressed", "interval entirely below zero"
    elif lo > 0:
        v, r = "improved", "interval entirely above zero"
    else:
        v, r = "inconclusive", "interval contains zero"
    return Comparison(k, len(diffs), d, lo, hi, es, v, r)
