"""Importer for exported OpenAI Evals (adapter module for that format).

Expected export layout (documented in docs/importer.md; reconstructed from the public Evals
API object shapes, not from a vendor-published file spec):

  EXPORT_DIR/eval.json    {"name", "data_source_config", "testing_criteria": [ ... ]}
  EXPORT_DIR/data.jsonl   one item per line: {"item": {"input": ..., "ideal"|"expected": ...}}
                          (bare {"input","ideal"} lines are accepted too)

Output: Inspect `dataset.jsonl` + `task.py` + `import_report.md` listing everything that was
not converted. Unconvertible graders are never silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCORER_FOR_STRING_CHECK = {"eq": "exact()", "ne": None, "like": "includes()", "ilike": "includes(ignore_case=True)"}


@dataclass
class ImportResult:
    items: int = 0
    scorers: list[str] = field(default_factory=list)
    not_converted: list[str] = field(default_factory=list)


def _input_text(inp: Any) -> str:
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        parts = []
        for m in inp:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return json.dumps(inp, sort_keys=True)


def import_openai(export_dir: str | Path, out_dir: str | Path) -> ImportResult:
    export, out = Path(export_dir), Path(out_dir)
    cfg = json.loads((export / "eval.json").read_text())
    res = ImportResult()
    out.mkdir(parents=True, exist_ok=True)

    lines = []
    for i, line in enumerate((export / "data.jsonl").read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        item = rec.get("item", rec)
        target = item.get("ideal", item.get("expected", item.get("target")))
        if target is None:
            res.not_converted.append(f"item {i}: no ideal/expected/target field; skipped")
            continue
        if isinstance(target, list):
            target = target[0] if target else ""
        extra = sorted(set(item) - {"input", "ideal", "expected", "target"})
        if extra:
            res.not_converted.append(f"item {i}: fields not carried over: {extra}")
        lines.append({"id": f"item-{i}", "input": _input_text(item.get("input")),
                      "target": str(target)})
    (out / "dataset.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    res.items = len(lines)

    for crit in cfg.get("testing_criteria", []):
        ctype, name = crit.get("type"), crit.get("name", crit.get("type"))
        if ctype == "string_check":
            op = crit.get("operation", "eq")
            scorer = SCORER_FOR_STRING_CHECK.get(op)
            if scorer:
                res.scorers.append(scorer)
            else:
                res.not_converted.append(f"grader {name!r}: string_check operation {op!r}")
            tmpl = str(crit.get("reference", ""))
            if tmpl and tmpl != "{{item.ideal}}" and tmpl != "{{item.expected}}":
                res.not_converted.append(
                    f"grader {name!r}: custom reference template {tmpl!r} replaced by dataset target"
                )
        else:
            res.not_converted.append(f"grader {name!r}: type {ctype!r} has no automatic mapping")
    if not res.scorers:
        res.scorers.append("exact()")
        res.not_converted.append("no convertible grader found; defaulted to exact()")

    task = (
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import json_dataset\n"
        "from inspect_ai.scorer import exact, includes\n"
        "from inspect_ai.solver import generate\n\n\n"
        "@task\n"
        "def imported():\n"
        "    return Task(\n"
        "        dataset=json_dataset(__file__.replace('task.py', 'dataset.jsonl')),\n"
        "        solver=generate(),\n"
        f"        scorer=[{', '.join(res.scorers)}],\n"
        "    )\n"
    )
    (out / "task.py").write_text(task)

    report = [f"# Import report: {cfg.get('name', 'eval')}", "",
              f"- Items converted: {res.items}",
              f"- Scorers emitted: {', '.join(res.scorers)}", "",
              "## Not converted", ""]
    report += [f"- {x}" for x in res.not_converted] or ["- (nothing)"]
    (out / "import_report.md").write_text("\n".join(report) + "\n")
    return res
