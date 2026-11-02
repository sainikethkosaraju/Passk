"""Importer for OpenAI Evals into runnable Inspect tasks (adapter module for those formats).

Two sources are understood:
  * Evals API export  : EXPORT/eval.json {name, testing_criteria[...]} + EXPORT/data.jsonl
                        ({"item": {"input", "ideal"|"expected"|...}} per line).
  * openai/evals repo : EXPORT/samples.jsonl ({"input": [...messages], "ideal": str|[str]}) plus
                        optional EXPORT/eval.yaml registry entry (class: ...Match/Includes/FuzzyMatch).

Output: dataset.jsonl, task.py (a real Inspect task using scorers in `passk.scorers`) and
import_report.md. EVERY grader or field that is not converted, or converted approximately, is
listed; nothing is dropped silently. `tests/test_importer.py` runs the generated task in Inspect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# string_check operation -> (scorer expression, exact?)
STRING_OPS = {"eq": ("exact()", True), "like": ("includes()", True),
              "ilike": ("includes(ignore_case=True)", True)}
LEGACY_CLASSES = {"match": ("match()", True), "includes": ("includes()", True),
                  "fuzzy_match": ("fuzzy_match()", False), "exact": ("exact()", True)}


@dataclass
class ImportResult:
    items: int = 0
    scorers: list[str] = field(default_factory=list)
    not_converted: list[str] = field(default_factory=list)
    approximate: list[str] = field(default_factory=list)


def _input_text(inp: Any) -> str:
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in inp)
    return json.dumps(inp, sort_keys=True)


def _load_items(export: Path, res: ImportResult) -> list[dict[str, str]]:
    path = export / "data.jsonl" if (export / "data.jsonl").exists() else export / "samples.jsonl"
    items = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        item = rec.get("item", rec)
        target = item.get("ideal", item.get("expected", item.get("target")))
        if target is None:
            res.not_converted.append(f"item {i}: no ideal/expected/target field; skipped")
            continue
        if isinstance(target, list):
            if len(target) > 1:
                res.approximate.append(f"item {i}: {len(target)} acceptable answers; only the first is the target")
            target = target[0] if target else ""
        extra = sorted(set(item) - {"input", "ideal", "expected", "target"})
        if extra:
            res.not_converted.append(f"item {i}: fields not carried over: {extra}")
        items.append({"id": f"item-{i}", "input": _input_text(item.get("input")), "target": str(target)})
    return items


def import_openai(export_dir: str | Path, out_dir: str | Path) -> ImportResult:
    export, out = Path(export_dir), Path(out_dir)
    res = ImportResult()
    out.mkdir(parents=True, exist_ok=True)
    name = export.name
    if (export / "eval.json").exists():
        cfg = json.loads((export / "eval.json").read_text())
        name = cfg.get("name", name)
        for crit in cfg.get("testing_criteria", []):
            ctype, cname = crit.get("type"), crit.get("name", crit.get("type"))
            if ctype == "string_check":
                scorer = STRING_OPS.get(crit.get("operation", "eq"))
                if scorer:
                    res.scorers.append(scorer[0])
                else:
                    res.not_converted.append(f"grader {cname!r}: string_check operation {crit.get('operation')!r}")
                ref = str(crit.get("reference", ""))
                if ref and ref not in ("{{item.ideal}}", "{{item.expected}}"):
                    res.not_converted.append(f"grader {cname!r}: custom reference template {ref!r} replaced by dataset target")
            elif ctype == "text_similarity":
                res.scorers.append(f"fuzzy_match(threshold={float(crit.get('pass_threshold', 0.8))})")
                res.approximate.append(
                    f"grader {cname!r}: text_similarity metric {crit.get('evaluation_metric')!r} "
                    "approximated by difflib ratio (passk.scorers.fuzzy_match)")
            elif ctype == "label_model":
                res.scorers.append("model_graded_qa()")
                res.approximate.append(
                    f"grader {cname!r}: label_model mapped to Inspect model_graded_qa(); labels "
                    f"{crit.get('labels')} and passing_labels {crit.get('passing_labels')} are NOT carried over; "
                    "needs a grader model at run time")
            else:
                res.not_converted.append(f"grader {cname!r}: type {ctype!r} has no automatic mapping")
    elif (export / "eval.yaml").exists():
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("legacy openai/evals registry import needs PyYAML: pip install pyyaml") from e
        reg = yaml.safe_load((export / "eval.yaml").read_text()) or {}
        for key, val in reg.items():
            cls = str(val["class"]).split(":")[0].rsplit(".", 1)[-1].lower() \
                if isinstance(val, dict) and "class" in val else ""
            if cls in LEGACY_CLASSES:
                expr, exact = LEGACY_CLASSES[cls]
                res.scorers.append(expr)
                if not exact:
                    res.approximate.append(f"registry {key!r}: {cls} approximated by difflib ratio")
            elif cls:
                res.not_converted.append(f"registry {key!r}: eval class {cls!r} has no automatic mapping")
    items = _load_items(export, res)
    (out / "dataset.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    res.items = len(items)
    if not res.scorers:
        res.scorers.append("exact()")
        res.not_converted.append("no convertible grader found; defaulted to exact()")
    uses_passk = any(s.startswith("fuzzy_match") for s in res.scorers)
    imports = ["from inspect_ai import Task, task", "from inspect_ai.dataset import json_dataset",
               "from inspect_ai.scorer import exact, includes, match, model_graded_qa",
               "from inspect_ai.solver import generate"]
    if uses_passk:
        imports.append("from passk.scorers import fuzzy_match")
    task = ("\n".join(imports) + "\n\nfrom pathlib import Path\n\n\n@task\ndef imported():\n"
            "    return Task(\n"
            "        dataset=json_dataset(str(Path(__file__).with_name('dataset.jsonl'))),\n"
            "        solver=generate(),\n"
            f"        scorer=[{', '.join(res.scorers)}],\n    )\n")
    (out / "task.py").write_text(task)
    rep = [f"# Import report: {name}", "", f"- Items converted: {res.items}",
           f"- Scorers emitted: {', '.join(res.scorers)}", "", "## Converted approximately", ""]
    rep += [f"- {x}" for x in res.approximate] or ["- (nothing)"]
    rep += ["", "## Not converted", ""] + ([f"- {x}" for x in res.not_converted] or ["- (nothing)"])
    (out / "import_report.md").write_text("\n".join(rep) + "\n")
    return res
