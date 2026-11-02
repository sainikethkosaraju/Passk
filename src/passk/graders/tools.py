"""Tool-call (trajectory) checks over MCP-style calls: {"name": str, "arguments": dict}."""

from __future__ import annotations

from typing import Any


def _match_args(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(k in actual and actual[k] == v for k, v in expected.items())


def check_tool_calls(
    calls: list[dict[str, Any]],
    *,
    must_call: list[dict[str, Any]] | None = None,
    must_not_call: list[str] | None = None,
    ordered: bool = False,
    max_calls: int | None = None,
) -> dict[str, Any]:
    """must_call entries are {"name":..., "arguments": subset-match}."""
    failures: list[str] = []
    for banned in must_not_call or []:
        if any(c["name"] == banned for c in calls):
            failures.append(f"called forbidden tool {banned}")
    cursor = 0
    for want in must_call or []:
        idx = next(
            (i for i, c in enumerate(calls)
             if i >= (cursor if ordered else 0) and c["name"] == want["name"]
             and _match_args(c.get("arguments", {}), want.get("arguments", {}))),
            None,
        )
        if idx is None:
            failures.append(f"missing call {want['name']}")
        elif ordered:
            cursor = idx + 1
    if max_calls is not None and len(calls) > max_calls:
        failures.append(f"{len(calls)} calls exceeds max {max_calls}")
    return {"passed": not failures, "failures": failures}
