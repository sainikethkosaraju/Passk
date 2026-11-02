"""Record/replay of model calls. The key includes the trial index so independent trials
never collapse into one cached answer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def cassette_key(model: str, messages: list, tools: list | None, sampling: dict,
                 trial: int) -> str:
    blob = json.dumps(
        {"model": model, "messages": messages, "tools": tools or [], "sampling": sampling,
         "trial": trial},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class Cassette:
    """mode: 'replay' (miss raises), 'record' (miss calls fn and stores), 'auto'."""

    def __init__(self, path: str | Path, mode: str = "replay") -> None:
        if mode not in {"replay", "record", "auto"}:
            raise ValueError(mode)
        self.path, self.mode = Path(path), mode
        self.store: dict[str, Any] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                self.store[rec["key"]] = rec["response"]

    def call(self, model: str, messages: list, tools: list | None, sampling: dict, trial: int,
             fn: Callable[[], Any]) -> Any:
        key = cassette_key(model, messages, tools, sampling, trial)
        if key in self.store:
            return self.store[key]
        if self.mode == "replay":
            raise KeyError(f"cassette miss for key {key[:12]} (replay mode forbids live calls)")
        resp = fn()
        self.store[key] = resp
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "response": resp}, sort_keys=True) + "\n")
        return resp
