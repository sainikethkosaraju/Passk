"""Flag flaky tasks (0 < c < n) and zero-success tasks."""

from __future__ import annotations

from passk.logs import TrialTable


def flag_tasks(table: TrialTable, min_trials: int = 2) -> dict[str, list[str]]:
    flaky: list[str] = []
    zero: list[str] = []
    for tid, (n, c) in sorted(table.counts().items()):
        if n < min_trials:
            continue
        if c == 0:
            zero.append(tid)
        elif c < n:
            flaky.append(tid)
    return {"flaky": flaky, "zero_success": zero}
