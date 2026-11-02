"""Build the JSON report (schema_version 1)."""

from __future__ import annotations

import math
from typing import Any

from passk.cost import PriceBook, cost_summary
from passk.flaky import flag_tasks
from passk.logs import TrialTable
from passk.metrics import aggregate, wilson

SCHEMA_VERSION = 1


def _pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def _clean(o: Any) -> Any:
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def build_report(table: TrialTable, book: PriceBook | None = None, max_k: int = 8) -> dict:
    counts = table.counts()
    pairs = list(counts.values())
    top = min(max_k, max((n for n, _ in pairs), default=0))
    ks = list(range(1, top + 1))
    lat = [t.latency for t in table.trials if t.latency is not None]
    tasks = []
    for tid, (n, c) in sorted(counts.items()):
        lo, hi = wilson(c, n)
        tasks.append({"id": tid, "n": n, "successes": c, "pass_rate": c / n,
                      "wilson_lo": lo, "wilson_hi": hi})
    try:
        cost = cost_summary(table, book)
    except FileNotFoundError:
        cost = {"error": "no price snapshot"}
    report = {
        "schema_version": SCHEMA_VERSION,
        "models": table.models(),
        "trials": len(table.trials),
        "tasks_count": len(counts),
        "pass_hat_k": {str(k): aggregate(pairs, k, kind="hat") for k in ks},
        "pass_at_k": {str(k): aggregate(pairs, k, kind="at") for k in ks},
        "latency": {"median_s": _pctl(lat, 0.5), "p95_s": _pctl(lat, 0.95)},
        "cost": cost,
        "flags": flag_tasks(table),
        "tasks": tasks,
        "warnings": list(table.warnings),
    }
    return _clean(report)
