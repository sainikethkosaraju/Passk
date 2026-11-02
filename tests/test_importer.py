import json
import re
from pathlib import Path

import pytest

from passk.importers.openai_evals import import_openai

ROOT = Path(__file__).parent.parent / "examples"


def test_api_export_report_is_honest(tmp_path):
    res = import_openai(ROOT / "openai_export", tmp_path)
    rep = (tmp_path / "import_report.md").read_text()
    assert res.items == 2
    assert "text_similarity" in rep and "label_model" in rep and "'python'" in rep
    assert "difficulty" in rep and "no ideal" in rep
    ds = [json.loads(x) for x in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    assert ds[0]["input"] == "Answer briefly.\nCapital of France?" or ds[0]["input"] == "Capital of France?"
    compile((tmp_path / "task.py").read_text(), "task.py", "exec")


def test_legacy_registry_mapping(tmp_path):
    res = import_openai(ROOT / "openai_export_legacy", tmp_path)
    assert res.items == 2 and any("fuzzy" in s for s in res.scorers)
    assert any("2 acceptable answers" in a for a in res.approximate)


def test_generated_task_actually_runs_in_inspect(tmp_path):
    pytest.importorskip("inspect_ai")
    import importlib.util

    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput

    import_openai(ROOT / "openai_export_legacy", tmp_path)
    spec = importlib.util.spec_from_file_location("imported_task", tmp_path / "task.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def answer(messages, tools, tc, cfg):
        q = messages[-1].text
        return ModelOutput.from_content("mockllm/model", "Paris" if "France" in q else "Tokyo")
    log = inspect_eval(mod.imported(), model="mockllm/model", model_args={"custom_outputs": answer},
                       log_dir=str(tmp_path / "logs"), display="none")[0]
    assert log.status == "success" and len(log.samples) == 2
    assert all(next(iter(s.scores.values())).value == "C" for s in log.samples)
    assert re.search(r"fuzzy_match", (tmp_path / "task.py").read_text())
