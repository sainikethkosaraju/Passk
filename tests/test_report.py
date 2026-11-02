import json

from passk.cli import main
from passk.logs import Trial, TrialTable, read_logs
from passk.report import build_report, render_html


def test_report_schema_v2_golden(logs, tmp_path):
    out = tmp_path / "o"
    assert main(["report", str(logs["base"]), "--out", str(out), "--alias", "mockllm/model=openai/gpt-oss-120b"]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["schema_version"] == 2 and sorted(rep) == ["frontier", "models", "schema_version", "trials", "warnings"]
    m = rep["models"]["mockllm/model"]
    assert sorted(m) == ["bayes_pass_hat_k", "cost", "errors", "flags", "latency", "pass_at_k", "pass_hat_k",
                         "tasks", "tasks_count", "tokens", "trials"]
    assert sorted(m["pass_hat_k"]["1"]) == ["excluded", "hi", "lo", "tasks", "value"]
    assert sorted(m["tasks"][0]) == ["bayes_pass_hat_kmax", "id", "jeffreys_hi", "jeffreys_lo", "n", "pass_rate",
                                     "successes", "trials", "wilson_hi", "wilson_lo"]
    # numeric snapshot of the committed real logs (guards the estimators end to end)
    assert round(m["pass_hat_k"]["1"]["value"], 3) == 0.74 and round(m["pass_at_k"]["5"]["value"], 3) == 0.9
    assert m["pass_hat_k"]["5"]["value"] <= m["pass_hat_k"]["1"]["value"]
    assert m["cost"]["price_resolution"]["mockllm/model"].startswith("alias+median-of-")
    html = (out / "report.html").read_text()
    assert html.count("<svg") >= 2 and "Reliability by k" in html and "Pareto" in html and "t8" in html


def test_html_escapes_untrusted_ids():
    t = TrialTable([Trial("x", "<script>alert(1)</script>", 1, True, "m<b>"),
                    Trial("x", "<script>alert(1)</script>", 2, False, "m<b>")])
    html = render_html(build_report(t, None))
    assert "<script>alert" not in html and "&lt;script&gt;" in html and "m&lt;b&gt;" in html


def test_multi_model_sections_and_frontier(logs):
    t = read_logs(logs["base"])
    for tr in read_logs(logs["reg"]).trials:
        tr.model = "other/model"
        t.trials.append(tr)
    from passk.cost import PriceBook
    book = PriceBook.load()
    book.aliases.update({"mockllm/model": "openai/gpt-oss-120b", "other/model": "gpt-4o"})
    rep = build_report(t, book)
    assert set(rep["models"]) == {"mockllm/model", "other/model"}
    assert rep["frontier"]["non_dominated"] and len(rep["frontier"]["points"]) == 2
