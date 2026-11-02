import json
import warnings
import zipfile
from pathlib import Path

import pytest

from passk.cost import PriceBook, price
from passk.flaky import flag_tasks
from passk.logs import read_logs, score_to_success

EX = Path(__file__).parent.parent / "examples" / "logs"


def test_reads_sample_logs():
    t = read_logs(EX / "sample_a.json")
    assert len(t.trials) == 50 and len(t.tasks()) == 10
    assert t.trials[0].input_tokens >= 1000 and t.trials[0].cache_read_tokens == 400
    assert t.warnings == []


def test_unknown_fields_reported(tmp_path):
    log = {"eval": {"task": "x", "model": "m"}, "status": "success",
           "samples": [{"id": 1, "epoch": 1, "scores": {"s": {"value": 1.0}}, "weird": 1}]}
    p = tmp_path / "l.json"
    p.write_text(json.dumps(log))
    t = read_logs(p)
    assert any("weird" in w for w in t.warnings)


def test_eval_zip_and_errors(tmp_path):
    p = tmp_path / "l.eval"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("header.json", json.dumps({"eval": {"task": "x", "model": "m"},
                                              "status": "success"}))
        z.writestr("samples/a_epoch_1.json", json.dumps(
            {"id": "a", "epoch": 1, "scores": {"s": {"value": "C"}}}))
        z.writestr("samples/a_epoch_2.json", json.dumps(
            {"id": "a", "epoch": 2, "scores": {"s": {"value": "C"}}, "error": {"message": "x"}}))
    t = read_logs(p)
    assert t.counts() == {"a": (2, 1)}  # errored trial counts as failure


def test_score_semantics():
    assert score_to_success({"a": {"value": "C"}})
    assert not score_to_success({"a": {"value": "I"}})
    assert not score_to_success({"a": {"value": "C"}, "b": {"value": 0.5}})
    assert not score_to_success({})
    assert score_to_success({"a": {"value": {"x": 1, "y": True}}})


def test_flaky_flags():
    t = read_logs(EX / "sample_a.json")
    f = flag_tasks(t)
    assert "task-8" in f["zero_success"]
    assert "task-5" in f["flaky"]


def test_cost_hand_computed():
    book = PriceBook.load()
    # 1000 in (400 cached), 200 out at gpt-oss-120b placeholder prices
    exp = 600 * 1.5e-7 + 400 * 1.5e-7 + 200 * 6e-7
    u = {"input_tokens": 1000, "output_tokens": 200, "cache_read_tokens": 400}
    assert price(u, "gpt-oss-120b", book) == pytest.approx(exp)
    # cached tier
    exp2 = 600 * 1e-6 + 400 * 1e-7 + 200 * 4e-6
    assert price(u, "example-long-context", book) == pytest.approx(exp2)
    # long-context tier
    big = {"input_tokens": 300_000, "output_tokens": 1000, "cache_read_tokens": 100_000}
    exp3 = 200_000 * 2e-6 + 100_000 * 1e-7 + 1000 * 8e-6
    assert price(big, "example-long-context", book) == pytest.approx(exp3)
    assert price(u, "openai/gpt-oss-120b", book) == pytest.approx(exp)
    with pytest.raises(KeyError):
        price(u, "nope", book)


def test_stale_snapshot_warns(tmp_path):
    p = tmp_path / "prices_2020-01-01.json"
    p.write_text(json.dumps({"_meta": {"date": "2020-01-01"}, "m": {
        "input_cost_per_token": 1, "output_cost_per_token": 1}}))
    with pytest.warns(UserWarning, match="stale"):
        PriceBook.load(p)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PriceBook.load()  # bundled snapshot is recent
