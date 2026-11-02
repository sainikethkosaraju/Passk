"""JSON report, schema_version 2: one section per model."""

from __future__ import annotations

import math
from typing import Any

from passk import stats
from passk.cost import PriceBook, cost_summary, pareto_front
from passk.flaky import flag_tasks
from passk.logs import TrialTable
from passk.metrics import aggregate

SCHEMA_VERSION = 2


def _pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def _clean(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    return o


def model_section(table: TrialTable, model: str, book: PriceBook | None, max_k: int, reps: int) -> dict:
    sub = table.for_model(model)
    counts = sub.counts()
    pairs = list(counts.values())
    top = min(max_k, max((n for n, _ in pairs), default=0))
    ks = list(range(1, top + 1))
    lat = [t.latency for t in sub.trials if t.latency is not None]
    tasks = []
    for tid, (n, c) in sorted(counts.items()):
        lo, hi = stats.wilson(c, n)
        jl, jh = stats.jeffreys(c, n)
        post = stats.posterior_pass_hat(c, n, max(top, 1))
        tasks.append({"id": tid, "n": n, "successes": c, "pass_rate": c / n,
                      "wilson_lo": lo, "wilson_hi": hi, "jeffreys_lo": jl, "jeffreys_hi": jh,
                      "bayes_pass_hat_kmax": post["mean"],
                      "trials": [int(t.success) for t in sorted(
                          sub.trials_by_task()[tid], key=lambda t: t.epoch)]})
    try:
        cost = cost_summary(sub, book)
    except FileNotFoundError:
        cost = {"error": "no price snapshot"}
    errors = sum(t.status == "error" for t in sub.trials)
    return {
        "trials": len(sub.trials), "tasks_count": len(counts), "errors": errors,
        "pass_hat_k": {str(k): aggregate(pairs, k, kind="hat", reps=reps) for k in ks},
        "pass_at_k": {str(k): aggregate(pairs, k, kind="at", reps=reps) for k in ks},
        "bayes_pass_hat_k": {str(k): stats.mean([stats.posterior_pass_hat(c, n, k)["mean"]
                                                 for n, c in pairs]) for k in range(1, 11)},
        "latency": {"median_s": _pctl(lat, 0.5), "p95_s": _pctl(lat, 0.95)},
        "tokens": {"input": sum(t.input_tokens for t in sub.trials),
                   "output": sum(t.output_tokens for t in sub.trials),
                   "cache_read": sum(t.cache_read_tokens for t in sub.trials)},
        "cost": cost, "flags": flag_tasks(sub), "tasks": tasks,
    }


def build_report(table: TrialTable, book: PriceBook | None = None, max_k: int = 8,
                 reps: int = 500, frontier_k: int = 3) -> dict:
    models = {m: model_section(table, m, book, max_k, reps) for m in table.models()}
    pts = []
    for m, sec in models.items():
        c = sec["cost"].get("usd_per_success")
        ks = sec["pass_hat_k"]
        kk = str(min(frontier_k, max((int(k) for k in ks), default=1)))
        if c is not None and kk in ks and not math.isnan(ks[kk]["value"]):
            pts.append((m, c, ks[kk]["value"]))
    return _clean({
        "schema_version": SCHEMA_VERSION, "trials": len(table.trials),
        "models": models, "frontier": {"k": frontier_k, "non_dominated": pareto_front(pts),
                                       "points": [{"model": m, "usd_per_success": c, "pass_hat_k": r}
                                                  for m, c, r in pts]},
        "warnings": list(table.warnings)})
