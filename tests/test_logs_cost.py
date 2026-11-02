import json
import warnings
import zipfile

import pytest

from passk.cost import PriceBook, build_snapshot, cost_summary, pareto_front, price
from passk.flaky import flag_tasks
from passk.logs import read_logs, score_to_success


def test_reads_real_inspect_json_and_zstd_eval(logs):
    j = read_logs(logs["base"])
    e = read_logs(logs["eval"])
    for t in (j, e):
        assert len(t.trials) == 50 and len(t.tasks()) == 10 and t.warnings == []
        assert t.trials[0].input_tokens >= 40 and t.trials[0].cache_read_tokens == 8
        assert t.models() == ["mockllm/model"] and t.trials[0].run_id
    # the .eval member compression really is zstd (this is what zipfile alone cannot read)
    z = zipfile.ZipFile(next((logs["eval"]).glob("*.eval")))
    assert {i.compress_type for i in z.infolist()} == {93}


def test_unknown_fields_and_errors(tmp_path):
    log = {"eval": {"task": "x", "model": "m"}, "status": "success",
           "samples": [{"id": 1, "epoch": 1, "scores": {"s": {"value": 1.0}}, "weird": 1},
                       {"id": 1, "epoch": 2, "scores": {"s": {"value": "C"}}, "error": {"message": "x"}}]}
    p = tmp_path / "l.json"
    p.write_text(json.dumps(log))
    t = read_logs(p)
    assert any("weird" in w for w in t.warnings)
    assert t.counts() == {"1": (2, 1)}  # errored trial counts as failure


def test_score_semantics():
    assert score_to_success({"a": {"value": "C"}})
    assert not score_to_success({"a": {"value": "I"}})
    assert not score_to_success({"a": {"value": "C"}, "b": {"value": 0.5}})
    assert not score_to_success({})
    assert score_to_success({"a": {"value": {"x": 1, "y": True}}})


def test_flags_on_real_logs(logs):
    f = flag_tasks(read_logs(logs["reg"]))
    assert f["zero_success"] and f["flaky"]
    assert not set(f["zero_success"]) & set(f["flaky"])


# ---- cost ---------------------------------------------------------------------------------
FIX = {
    "sample_spec": {"mode": "chat"},
    "gpt-oss-120b": {"mode": "chat", "input_cost_per_token": 1e-7, "output_cost_per_token": 4e-7},
    "provA/gpt-oss-120b": {"mode": "chat", "input_cost_per_token": 2e-7, "output_cost_per_token": 6e-7},
    "provB/GPT-OSS-120B": {"mode": "chat", "input_cost_per_token": 6e-7, "output_cost_per_token": 8e-7},
    "long-ctx": {"mode": "chat", "input_cost_per_token": 1e-6, "output_cost_per_token": 4e-6,
                 "cache_read_input_token_cost": 1e-7, "input_cost_per_token_above_200k_tokens": 2e-6,
                 "output_cost_per_token_above_200k_tokens": 8e-6},
    "embed-model": {"mode": "embedding", "input_cost_per_token": 1e-9},
}


@pytest.fixture
def book(tmp_path):
    src = tmp_path / "src.json"
    src.write_text(json.dumps(FIX))
    import datetime as dt
    out = build_snapshot(src, tmp_path, include="gpt-oss|long", today=dt.date.today(),
                         aliases={"mock/model": "provA/gpt-oss-120b"})
    return PriceBook.load(out)


def test_snapshot_filters_and_records_provenance(book, tmp_path):
    assert set(book.models) == {"gpt-oss-120b", "provA/gpt-oss-120b", "provB/GPT-OSS-120B", "long-ctx"}
    assert len(book.sha256) == 64 and book.source.endswith("src.json")


def test_resolution_modes(book):
    assert book.resolve("gpt-oss-120b").how == "exact"
    assert book.resolve("mock/model").how == "alias"
    assert book.resolve("vendor/gpt-oss-120b").how == "stripped-prefix"
    assert book.resolve("other/gpt-oss-120B").how == "median-of-3-providers"  # case-insensitive base match
    med = PriceBook(book.date, {k: v for k, v in book.models.items() if "/" in k}, "t", {}).resolve("x/gpt-oss-120b")
    assert med.how == "median-of-2-providers" and med.entry["input_cost_per_token"] == pytest.approx(4e-7)
    with pytest.raises(KeyError):
        book.resolve("nope")


def test_cost_hand_computed(book):
    u = {"input_tokens": 1000, "output_tokens": 200, "cache_read_tokens": 400}
    assert price(u, "long-ctx", book) == pytest.approx(600 * 1e-6 + 400 * 1e-7 + 200 * 4e-6)
    big = {"input_tokens": 300_000, "output_tokens": 1000, "cache_read_tokens": 100_000}
    assert price(big, "long-ctx", book) == pytest.approx(200_000 * 2e-6 + 100_000 * 1e-7 + 1000 * 8e-6)
    assert price(u, "gpt-oss-120b", book) == pytest.approx(1000 * 1e-7 + 200 * 4e-7)


def test_cost_summary_and_stale(book, logs, tmp_path):
    book.aliases["mockllm/model"] = "gpt-oss-120b"
    s = cost_summary(read_logs(logs["base"]), book)
    assert s["total_usd"] > 0 and s["usd_per_success"] == pytest.approx(s["total_usd"] / s["successes"])
    assert s["price_resolution"] == {"mockllm/model": "alias"} and s["missing_prices"] == []
    lo, hi = s["usd_per_success_ci"]
    assert lo <= s["usd_per_success"] <= hi
    assert cost_summary(read_logs(logs["base"]), PriceBook(book.date, {}, "x", {}))["missing_prices"]
    p = tmp_path / "prices_2020-01-01.json"
    p.write_text(json.dumps({"_meta": {"date": "2020-01-01"}, "m": {"input_cost_per_token": 1, "output_cost_per_token": 1}}))
    with pytest.warns(UserWarning, match="stale"):
        PriceBook.load(p)


def test_bundled_snapshot_is_real_and_hashed():
    b = PriceBook.load()
    assert len(b.models) > 1000 and len(b.sha256) == 64 and "litellm" in b.source.lower() or "model_prices" in b.source
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert b.resolve("openai/gpt-oss-120b").entry["input_cost_per_token"] > 0


def test_pareto():
    pts = [("a", 1.0, 0.5), ("b", 2.0, 0.9), ("c", 3.0, 0.8), ("d", 0.5, 0.2)]
    assert sorted(pareto_front(pts)) == ["a", "b", "d"]
