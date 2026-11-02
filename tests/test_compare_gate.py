import json
import random

from passk.cli import main
from passk.compare import compare, compare_ks
from passk.logs import Trial, TrialTable, read_logs


def table_from(rates, n, seed, model="m"):
    rng = random.Random(seed)
    t = TrialTable()
    for i, p in enumerate(rates):
        for e in range(n):
            t.trials.append(Trial("x", f"t{i}", e + 1, rng.random() < p, model))
    return t


def test_detects_injected_regression_on_real_logs(logs):
    c = compare(read_logs(logs["base"]), read_logs(logs["reg"]), 1)
    assert c.verdict == "regressed" and c.delta < -0.3 and c.hi < 0 and c.p_value < 0.05
    assert c.losses > c.wins and c.mde > 0


def test_noise_is_inconclusive_on_real_logs(logs):
    c = compare(read_logs(logs["base"]), read_logs(logs["eval"]), 1)
    assert c.verdict == "inconclusive" and "MDE" in c.reason or "detect" in c.reason


def test_refuses_below_min_trials(logs):
    c = compare(read_logs(logs["base"]), read_logs(logs["reg"]), 1, min_trials=6)
    assert c.verdict == "inconclusive" and "refusing" in c.reason


def test_type_one_error_under_the_null_is_controlled():
    """Same underlying reliabilities on both sides: verdict must be non-inconclusive rarely."""
    rng = random.Random(0)
    rates = [rng.betavariate(2, 1) for _ in range(30)]
    false_calls, sims = 0, 60
    for s in range(sims):
        a, b = table_from(rates, 6, 2 * s), table_from(rates, 6, 2 * s + 1)
        false_calls += compare(a, b, 1, reps=150, seed=s).verdict != "inconclusive"
    assert false_calls / sims <= 0.12  # nominal 0.05; allow Monte-Carlo slack


def test_power_to_detect_a_real_drop():
    rng = random.Random(1)
    rates = [rng.betavariate(3, 1) for _ in range(30)]
    worse = [max(0.0, p - 0.25) for p in rates]
    hits, sims = 0, 40
    for s in range(sims):
        hits += compare(table_from(rates, 6, 2 * s), table_from(worse, 6, 2 * s + 1), 1,
                        reps=150, seed=s).verdict == "regressed"
    assert hits / sims >= 0.9


def test_holm_across_k(logs):
    res = compare_ks(read_logs(logs["base"]), read_logs(logs["reg"]), [1, 2, 3, 5])
    assert all(c.p_adjusted is not None and c.p_adjusted >= c.p_value for c in res)


def test_gate_modes_and_exit_codes(logs, capsys):
    b, r, e = (str(logs[k]) for k in ("base", "reg", "eval"))
    assert main(["gate", "--baseline", b, "--candidate", r]) == 1
    assert main(["gate", "--baseline", b, "--candidate", e]) == 0
    assert main(["gate", "--baseline", b, "--candidate", e, "--mode", "point", "--max-drop", "0.01"]) == 1
    assert main(["gate", "--baseline", b, "--candidate", e, "--mode", "noninferior", "--max-drop", "0.05"]) == 1
    assert main(["gate", "--baseline", b, "--candidate", r, "--format", "markdown"]) == 1
    out = capsys.readouterr().out
    assert "### Passk gate: **FAIL**" in out and "| pass^1 |" in out
    assert main(["gate", "--baseline", b, "--candidate", r, "--min-trials", "9", "--mode", "noninferior",
                 "--strict"]) == 2
    main(["gate", "--baseline", b, "--candidate", r, "--format", "json"])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "FAIL"


def test_compare_power_and_prices_commands(logs, capsys):
    assert main(["compare", str(logs["base"]), str(logs["reg"]), "-k", "1", "-k", "3"]) == 0
    assert "Holm" in capsys.readouterr().out
    assert main(["power", "--delta", "0.1", "--sd", "0.2"]) == 0
    assert "tasks needed" in capsys.readouterr().out
