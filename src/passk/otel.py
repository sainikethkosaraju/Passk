"""The single adapter for OpenTelemetry GenAI semantic conventions (status: Development).

Spans are emitted as plain dicts / JSONL so no OpenTelemetry dependency is needed; an OTLP
export is available if the `otlp` extra is installed. Attribute names are isolated here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from passk.logs import TrialTable


def trials_to_spans(table: TrialTable) -> list[dict[str, Any]]:
    spans = []
    for t in table.trials:
        spans.append({
            "name": f"chat {t.model}",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": t.model,
                "gen_ai.usage.input_tokens": t.input_tokens,
                "gen_ai.usage.output_tokens": t.output_tokens,
                "passk.task": t.sample_id,
                "passk.epoch": t.epoch,
                "passk.success": t.success,
                "passk.latency_s": t.latency,
            },
            "status": "OK" if t.status == "ok" else "ERROR",
        })
    return spans


def write_jsonl(spans: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in spans))


def export_otlp(spans: list[dict[str, Any]], endpoint: str) -> int:
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("install passk[otlp] to export over OTLP") from e
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    tracer = provider.get_tracer("passk")
    for s in spans:  # pragma: no cover
        with tracer.start_as_current_span(s["name"], attributes={
            k: v for k, v in s["attributes"].items() if v is not None
        }):
            pass
    provider.shutdown()  # pragma: no cover
    _ = trace
    return len(spans)  # pragma: no cover
