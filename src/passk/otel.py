"""The single adapter for OpenTelemetry GenAI semantic conventions (status: Development).

All attribute names live in `ATTR`. `trials_to_spans` is dependency-free (plain dicts / JSONL);
`export_spans` emits real OTel spans through the SDK to any `SpanExporter` (OTLP/HTTP, console,
in-memory for tests) with the trial's recorded timing as span start/end."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from passk.logs import TrialTable

ATTR = {
    "op": "gen_ai.operation.name", "model": "gen_ai.request.model",
    "in_tok": "gen_ai.usage.input_tokens", "out_tok": "gen_ai.usage.output_tokens",
    "task": "passk.task", "epoch": "passk.epoch", "success": "passk.success",
    "latency": "passk.latency_s", "run": "passk.run_id",
}


def trials_to_spans(table: TrialTable) -> list[dict[str, Any]]:
    spans = []
    for t in table.trials:
        spans.append({
            "name": f"chat {t.model}",
            "attributes": {ATTR["op"]: "chat", ATTR["model"]: t.model, ATTR["in_tok"]: t.input_tokens,
                           ATTR["out_tok"]: t.output_tokens, ATTR["task"]: t.sample_id,
                           ATTR["epoch"]: t.epoch, ATTR["success"]: t.success,
                           ATTR["latency"]: t.latency, ATTR["run"]: t.run_id},
            "status": "OK" if t.status == "ok" else "ERROR"})
    return spans


def write_jsonl(spans: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in spans))


def export_spans(spans: list[dict[str, Any]], exporter: Any) -> int:
    """Emit spans via the OpenTelemetry SDK. `exporter` is any SpanExporter instance."""
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.trace import Status, StatusCode
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("install passk[otlp]") from e
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("passk")
    now = time.time_ns()
    for s in spans:
        dur = int((s["attributes"].get(ATTR["latency"]) or 0) * 1e9)
        span = tracer.start_span(s["name"], start_time=now - dur,
                                 attributes={k: v for k, v in s["attributes"].items() if v is not None})
        span.set_status(Status(StatusCode.OK if s["status"] == "OK" else StatusCode.ERROR))
        span.end(end_time=now)
    provider.shutdown()
    return len(spans)


def export_otlp(spans: list[dict[str, Any]], endpoint: str) -> int:  # pragma: no cover - network
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
        OTLPSpanExporter,
    )
    return export_spans(spans, OTLPSpanExporter(endpoint=endpoint))
