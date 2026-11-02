"""Inspect AI integration (optional; `pip install passk[inspect]`). Registered via the
`inspect_ai` entry point so it is available as soon as passk is installed.

1. Hook: with `PASSK_REPORT=1`, every finished task writes `<log_dir>/passk/<task>.json|.html`
   next to the Inspect log (reliability with intervals, cost, flaky tasks).
2. Model provider `passk`: `passk/record/<inner model>` records every generation to a cassette
   directory; `passk/replay/<inner model>` serves them with NO inner model and NO network. The
   cassette key includes the epoch (trial index), so independent trials never collapse into one
   cached answer. `PASSK_CASSETTE_DIR` (default `./cassettes`) selects the directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from inspect_ai.hooks import Hooks, TaskEnd, hooks
from inspect_ai.log import EvalLog
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    get_model,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo


def evallog_to_dict(log: EvalLog) -> dict[str, Any]:
    """Reduce an in-memory EvalLog to the dict shape Passk's reader consumes. Only the needed
    fields are extracted (never a full model_dump: eval specs may hold non-serialisable objects)."""
    samples = []
    for s in log.samples or []:
        samples.append({
            "id": s.id, "epoch": s.epoch,
            "scores": {k: {"value": v.value} for k, v in (s.scores or {}).items()},
            "model_usage": {m: u.model_dump(exclude_none=True) for m, u in (s.model_usage or {}).items()},
            "total_time": s.total_time,
            "error": bool(s.error),
            "limit": {"type": s.limit.type} if s.limit is not None else None,
            "messages": [None] * len(s.messages or []),
        })
    return {"status": log.status,
            "eval": {"task": log.eval.task, "model": log.eval.model, "run_id": log.eval.run_id},
            "samples": samples}


@hooks(name="passk", description="Write a Passk reliability/cost report after each task (PASSK_REPORT=1).")
class PasskHooks(Hooks):
    def enabled(self) -> bool:
        return os.environ.get("PASSK_REPORT", "") not in ("", "0", "false")

    async def on_task_end(self, data: TaskEnd) -> None:
        from passk.logs import TrialTable, read_log_dict
        from passk.report import build_report, render_html

        log = data.log
        if not log.samples:
            return
        table = TrialTable()
        read_log_dict(evallog_to_dict(log), table, name=log.eval.task)
        rep = build_report(table, None, reps=300)
        loc = Path(str(log.location).replace("file://", "")) if log.location else Path("logs/x")
        out = loc.parent / "passk"
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{log.eval.task}-{data.eval_id[:8]}"
        (out / f"{stem}.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
        (out / f"{stem}.html").write_text(render_html(rep))


def _active_epoch() -> int:
    from inspect_ai.log._samples import sample_active
    s = sample_active()
    return int(s.epoch) if s is not None else 0


def cassette_key(model: str, messages: list[Any], tools: list[Any], tool_choice: Any,
                 config: Any, epoch: int) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "tools": tools, "tc": tool_choice,
                       "cfg": config, "epoch": epoch}, sort_keys=True, default=str,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


@modelapi(name="passk")
class PasskCassetteAPI(ModelAPI):
    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 config: GenerateConfig = GenerateConfig(), **model_args: Any) -> None:
        super().__init__(model_name, base_url, api_key, [], config)
        mode, _, inner = model_name.partition("/")
        if mode not in {"record", "replay"} or not inner:
            raise ValueError("use passk/record/<model> or passk/replay/<model>")
        self.mode, self.inner_name = mode, inner
        self.model_args = model_args
        self.dir = Path(os.environ.get("PASSK_CASSETTE_DIR", "cassettes"))
        self._inner: Model | None = None

    def _get_inner(self) -> Model:
        if self._inner is None:
            self._inner = get_model(self.inner_name, **self.model_args)
        return self._inner

    async def generate(self, input: list[ChatMessage], tools: list[ToolInfo],
                       tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
        key = cassette_key(
            self.inner_name, [m.model_dump(mode="json", exclude={"id"}) for m in input],
            [t.model_dump(mode="json") for t in tools],
            tool_choice if isinstance(tool_choice, str) else str(tool_choice),
            config.model_dump(mode="json", exclude_none=True), _active_epoch())
        path = self.dir / f"{key}.json"
        if path.exists():
            return ModelOutput.model_validate_json(path.read_text())
        if self.mode == "replay":
            raise RuntimeError(f"cassette miss ({key[:12]}): replay mode never calls a live model")
        out = await self._get_inner().generate(input, tools, tool_choice, config)
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(out.model_dump_json())
        return out
