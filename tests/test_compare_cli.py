import json
from pathlib import Path

from passk.cli import main
from passk.compare import compare
from passk.logs import read_logs

EX = Path(__file__).parent.parent / "examples" / "logs"


def test_detects_injected_drop():
    c = compare(read_logs(EX / "sample_a.json"), read_logs(EX / "sample_b_regressed.json"))
    assert c.verdict == "regressed" and c.delta < -0.2


def test_noise_is_inconclusive():
    c = compare(read_logs(EX / "sample_a.json"), read_logs(EX / "sample_a_rerun.json"))
    assert c.verdict == "inconclusive"


def test_refuses_below_min_trials():
    c = compare(read_logs(EX / "sample_a.json"), read_logs(EX / "sample_b_regressed.json"),
                min_trials=6)
    assert c.verdict == "inconclusive" and "refusing" in c.reason


def test_gate_exit_codes(capsys):
    a, b, r = (str(EX / n) for n in ("sample_a.json", "sample_a_rerun.json",
                                     "sample_b_regressed.json"))
    assert main(["gate", "--baseline", a, "--candidate", r, "--max-drop", "0.05"]) == 1
    assert main(["gate", "--baseline", a, "--candidate", b, "--max-drop", "0.15"]) == 0


def test_report_json_and_html_golden(tmp_path, capsys):
    out = tmp_path / "o"
    assert main(["report", str(EX / "sample_a.json"), "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["schema_version"] == 1
    assert sorted(rep) == ["cost", "flags", "latency", "models", "pass_at_k", "pass_hat_k",
                           "schema_version", "tasks", "tasks_count", "trials", "warnings"]
    assert sorted(rep["pass_hat_k"]["1"]) == ["excluded", "hi", "lo", "tasks", "value"]
    assert sorted(rep["tasks"][0]) == ["id", "n", "pass_rate", "successes", "wilson_hi",
                                       "wilson_lo"]
    assert sorted(rep["cost"]) == ["missing_prices", "price_snapshot", "price_source",
                                   "successes", "total_usd", "trials", "usd_per_success"]
    html = (out / "report.html").read_text()
    assert "<svg" in html and "task-0" in html and "pass^k" in html
    assert rep["pass_hat_k"]["5"]["value"] <= rep["pass_hat_k"]["1"]["value"]


def test_cost_cmd_and_summary(capsys):
    assert main(["cost", str(EX / "sample_a.json")]) == 0
    assert main(["report", str(EX / "sample_a.json")]) == 0
    assert "pass^1" in capsys.readouterr().out
