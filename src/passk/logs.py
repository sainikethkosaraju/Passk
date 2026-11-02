"""Inspect AI log reader: .json logs and .eval (zip) logs into a flat trial table.

Unknown sample fields are reported in `TrialTable.warnings`, never silently dropped.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWN_SAMPLE_KEYS = {
    "id", "epoch", "input", "target", "messages", "output", "scores", "metadata", "store",
    "events", "model_usage", "role_usage", "started_at", "completed_at", "total_time",
    "working_time", "uuid", "error", "error_retries", "attachments", "events_data",
    "turn_count", "limit", "sandbox", "files", "setup", "choices",
}
SUCCESS_STRINGS = {"C", "CORRECT", "PASS", "TRUE", "YES"}


@dataclass
class Trial:
    task: str
    sample_id: str
    epoch: int
    success: bool
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    latency: float | None = None
    status: str = "ok"  # ok | error


@dataclass
class TrialTable:
    trials: list[Trial] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def tasks(self) -> list[str]:
        return sorted({t.sample_id for t in self.trials})

    def counts(self) -> dict[str, tuple[int, int]]:
        """sample_id -> (n_trials, n_successes). Errored trials count as failures."""
        out: dict[str, list[int]] = {}
        for t in self.trials:
            n, c = out.setdefault(t.sample_id, [0, 0])
            out[t.sample_id] = [n + 1, c + int(t.success)]
        return {k: (v[0], v[1]) for k, v in out.items()}

    def models(self) -> list[str]:
        return sorted({t.model for t in self.trials})


def score_to_success(scores: dict[str, Any] | None) -> bool:
    """A trial succeeds if every score is correct (value 'C', True, or numeric >= 1)."""
    if not scores:
        return False
    for s in scores.values():
        v = s.get("value") if isinstance(s, dict) else s
        if isinstance(v, dict):  # multi-metric score: all components must pass
            if not all(_value_ok(x) for x in v.values()):
                return False
        elif not _value_ok(v):
            return False
    return True


def _value_ok(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int | float):
        return v >= 1
    if isinstance(v, str):
        return v.strip().upper() in SUCCESS_STRINGS
    return False


def _load_raw(path: Path) -> dict[str, Any]:
    if path.suffix == ".eval":
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            header = json.loads(z.read("header.json")) if "header.json" in names else {}
            samples = []
            for n in sorted(x for x in names if x.startswith("samples/") and x.endswith(".json")):
                samples.append(json.loads(z.read(n)))
            header["samples"] = samples
            return header
    return json.loads(path.read_text())


def read_log(path: str | Path, table: TrialTable | None = None) -> TrialTable:
    path = Path(path)
    table = table or TrialTable()
    raw = _load_raw(path)
    ev = raw.get("eval", {}) or {}
    model = ev.get("model", "unknown")
    unknown: set[str] = set()
    for s in raw.get("samples", []) or []:
        unknown |= set(s) - KNOWN_SAMPLE_KEYS
        usage = (s.get("model_usage") or {})
        inp = out = cache = reason = 0
        for u in usage.values():
            inp += int(u.get("input_tokens", 0) or 0)
            out += int(u.get("output_tokens", 0) or 0)
            cache += int(u.get("input_tokens_cache_read", 0) or 0)
            reason += int(u.get("reasoning_tokens", 0) or 0)
        errored = bool(s.get("error"))
        table.trials.append(
            Trial(
                task=ev.get("task", path.stem),
                sample_id=str(s.get("id")),
                epoch=int(s.get("epoch", 1)),
                success=(not errored) and score_to_success(s.get("scores")),
                model=model,
                input_tokens=inp,
                output_tokens=out,
                cache_read_tokens=cache,
                reasoning_tokens=reason,
                latency=s.get("total_time"),
                status="error" if errored else "ok",
            )
        )
    if unknown:
        table.warnings.append(f"{path.name}: unknown sample fields: {sorted(unknown)}")
    if raw.get("status") not in (None, "success"):
        table.warnings.append(f"{path.name}: log status is {raw.get('status')!r}")
    return table


def read_logs(*paths: str | Path) -> TrialTable:
    """Read files or directories (recursively) of .json / .eval logs."""
    table = TrialTable()
    for p in paths:
        p = Path(p)
        files = sorted([*p.rglob("*.json"), *p.rglob("*.eval")]) if p.is_dir() else [p]
        for f in files:
            read_log(f, table)
    if not table.trials:
        table.warnings.append("no trials found")
    return table
