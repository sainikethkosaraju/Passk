import json
from pathlib import Path

import pytest

from passk.cassette import Cassette, cassette_key
from passk.graders import calibrate, check_state, check_tool_calls, model_grade
from passk.importers.openai_evals import import_openai
from passk.logs import read_logs
from passk.otel import trials_to_spans, write_jsonl
from passk.pacing import BudgetExhausted, DailyBudget, TokenBucket

ROOT = Path(__file__).parent.parent


def test_importer(tmp_path):
    res = import_openai(ROOT / "examples" / "openai_export", tmp_path)
    assert res.items == 2
    rep = (tmp_path / "import_report.md").read_text()
    assert "label_model" in rep or "tone" in rep
    assert "difficulty" in rep and "no ideal" in rep
    ds = [json.loads(x) for x in (tmp_path / "dataset.jsonl").read_text().splitlines()]
    assert ds[0] == {"id": "item-0", "input": "Capital of France?", "target": "Paris"}
    compile((tmp_path / "task.py").read_text(), "task.py", "exec")


def test_cassette_replay_and_independent_trials(tmp_path):
    c = Cassette(tmp_path / "c.jsonl", "record")
    calls = []
    def live(i):
        calls.append(i); return f"out-{i}"
    msgs = [{"role": "user", "content": "hi"}]
    outs = [c.call("m", msgs, None, {"t": 0.5}, t, lambda t=t: live(t)) for t in range(3)]
    assert outs == ["out-0", "out-1", "out-2"]  # trials do not collapse
    r = Cassette(tmp_path / "c.jsonl", "replay")
    assert [r.call("m", msgs, None, {"t": 0.5}, t, lambda: 1 / 0) for t in range(3)] == outs
    with pytest.raises(KeyError):
        r.call("m", msgs, None, {"t": 0.5}, 9, lambda: "x")
    assert cassette_key("m", msgs, None, {}, 0) != cassette_key("m", msgs, None, {}, 1)


def test_graders():
    assert check_state({"flag": True, "n": 2}, [("f", lambda s: s["flag"]),
                                                ("n", lambda s: s["n"] == 2)])["passed"]
    assert not check_state({}, [("boom", lambda s: s["x"])])["passed"]
    calls = [{"name": "search", "arguments": {"q": "a"}}, {"name": "apply", "arguments": {}}]
    assert check_tool_calls(calls, must_call=[{"name": "search", "arguments": {"q": "a"}},
                                              {"name": "apply"}], ordered=True)["passed"]
    assert not check_tool_calls(calls, must_call=[{"name": "apply"}, {"name": "search"}],
                                ordered=True)["passed"]
    assert not check_tool_calls(calls, must_not_call=["apply"])["passed"]
    assert not check_tool_calls(calls, max_calls=1)["passed"]
    assert model_grade(lambda p: "PASS.", "q", "a", "r") == "pass"
    assert model_grade(lambda p: "I think so", "q", "a", "r") == "unknown"
    labeled = [{"question": "q", "answer": "a", "rubric": "r", "label": "pass"},
               {"question": "q", "answer": "b", "rubric": "r", "label": "fail"},
               {"question": "q", "answer": "c", "rubric": "r", "label": "fail"}]
    answers = iter(["pass", "pass", "unknown"])
    cal = calibrate(lambda p: next(answers), labeled)
    assert cal["unknown"] == 1 and cal["fp"] == 1 and cal["tp"] == 1


def test_otel_and_pacing(tmp_path):
    t = read_logs(ROOT / "examples" / "logs" / "sample_a.json")
    spans = trials_to_spans(t)
    assert spans[0]["attributes"]["gen_ai.operation.name"] == "chat"
    write_jsonl(spans, tmp_path / "s.jsonl")
    assert len((tmp_path / "s.jsonl").read_text().splitlines()) == 50
    now = [0.0]; slept = []
    def sleep(s): slept.append(s); now[0] += s
    b = TokenBucket(60, burst=1, clock=lambda: now[0], sleep=sleep)
    b.acquire(); b.acquire()
    assert slept and abs(slept[0] - 1.0) < 1e-9
    d = DailyBudget(2); d.spend(); d.spend()
    with pytest.raises(BudgetExhausted):
        d.spend()
