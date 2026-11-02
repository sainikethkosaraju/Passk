import pytest

from passk.cassette import Cassette, cassette_key
from passk.graders import calibrate, check_state, check_tool_calls, model_grade
from passk.logs import read_logs
from passk.otel import ATTR, export_spans, trials_to_spans, write_jsonl
from passk.pacing import BudgetExhausted, DailyBudget, TokenBucket


def test_cassette_independent_trials(tmp_path):
    c = Cassette(tmp_path / "c.jsonl", "record")
    msgs = [{"role": "user", "content": "hi"}]
    outs = [c.call("m", msgs, None, {"t": 0.5}, t, lambda t=t: f"out-{t}") for t in range(3)]
    r = Cassette(tmp_path / "c.jsonl", "replay")
    assert [r.call("m", msgs, None, {"t": 0.5}, t, lambda: 1 / 0) for t in range(3)] == outs
    with pytest.raises(KeyError):
        r.call("m", msgs, None, {"t": 0.5}, 9, lambda: "x")
    assert cassette_key("m", msgs, None, {}, 0) != cassette_key("m", msgs, None, {}, 1)


def test_graders():
    assert check_state({"flag": True}, [("f", lambda s: s["flag"])])["passed"]
    assert not check_state({}, [("boom", lambda s: s["x"])])["passed"]
    calls = [{"name": "search", "arguments": {"q": "a"}}, {"name": "apply", "arguments": {}}]
    assert check_tool_calls(calls, must_call=[{"name": "search", "arguments": {"q": "a"}}, {"name": "apply"}],
                            ordered=True)["passed"]
    assert not check_tool_calls(calls, must_call=[{"name": "apply"}, {"name": "search"}], ordered=True)["passed"]
    assert not check_tool_calls(calls, must_not_call=["apply"])["passed"]
    assert model_grade(lambda p: "PASS.", "q", "a", "r") == "pass"
    assert model_grade(lambda p: "I think so", "q", "a", "r") == "unknown"
    ans = iter(["pass", "pass", "unknown"])
    cal = calibrate(lambda p: next(ans), [{"question": "q", "answer": a, "rubric": "r", "label": lab}
                                          for a, lab in (("a", "pass"), ("b", "fail"), ("c", "fail"))])
    assert cal["unknown"] == 1 and cal["fp"] == 1 and cal["tp"] == 1


def test_otel_real_sdk_export(logs, tmp_path):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    t = read_logs(logs["base"])
    spans = trials_to_spans(t)
    exp = InMemorySpanExporter()
    assert export_spans(spans, exp) == 50
    got = exp.get_finished_spans()
    assert len(got) == 50 and got[0].attributes[ATTR["op"]] == "chat"
    assert got[0].attributes[ATTR["model"]] == "mockllm/model" and ATTR["success"] in got[0].attributes
    assert all(s.end_time >= s.start_time for s in got)
    write_jsonl(spans, tmp_path / "s.jsonl")
    assert len((tmp_path / "s.jsonl").read_text().splitlines()) == 50


def test_pacing():
    now = [0.0]; slept = []
    def sleep(s): slept.append(s); now[0] += s
    b = TokenBucket(60, burst=1, clock=lambda: now[0], sleep=sleep)
    b.acquire(); b.acquire()
    assert slept and abs(slept[0] - 1.0) < 1e-9
    d = DailyBudget(2); d.spend(); d.spend()
    with pytest.raises(BudgetExhausted):
        d.spend()
