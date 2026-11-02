"""State checks: grade the outcome in the store, not the path."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Check = tuple[str, Callable[[dict[str, Any]], bool]]


def check_state(state: dict[str, Any], checks: list[Check]) -> dict[str, Any]:
    """Run named predicates over final state. A raising predicate counts as a failure."""
    results = {}
    for name, fn in checks:
        try:
            results[name] = bool(fn(state))
        except Exception:
            results[name] = False
    return {"passed": all(results.values()), "checks": results}
