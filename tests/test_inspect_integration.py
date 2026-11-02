"""These tests run REAL Inspect (mockllm provider; no network, no cost)."""
import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("inspect_ai")
from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import ModelOutput  # noqa: E402
from inspect_ai.scorer import Score, pass_at, pass_k  # noqa: E402

from passk.logs import read_logs  # noqa: E402
from passk.metrics import pass_at_k, pass_hat_k  # noqa: E402


def test_estimators_match_inspects_own_reducers_on_real_logs(logs):
    """Cross-validation against upstream: per-sample pass^k / pass@k equal Inspect's reducers."""
    for key in ("base", "reg", "eval"):
        path = next(logs[key].glob("*"))
        log = read_eval_log(str(path))
        table = read_logs(logs[key])
        counts = table.counts()
        by_id: dict = {}
        for s in log.samples:
            by_id.setdefault(str(s.id), []).append(next(iter(s.scores.values())))
        for k in (1, 2, 3, 5):
            for sid, scores in by_id.items():
                n, c = counts[sid]
                ours_hat, ours_at = pass_hat_k(n, c, k), pass_at_k(n, c, k)
                theirs_hat = pass_k(k)([Score(value=sc.value) for sc in scores]).value
                theirs_at = pass_at(k)([Score(value=sc.value) for sc in scores]).value
                assert ours_hat == pytest.approx(theirs_hat), (key, sid, k)
                assert ours_at == pytest.approx(theirs_at), (key, sid, k)


def test_passk_reads_what_inspect_just_wrote(tmp_path):
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate

    @task
    def tiny():
        return Task(dataset=[Sample(input="What is 1+1?", target="2", id="a"),
                             Sample(input="What is 2+2?", target="4", id="b")],
                    solver=generate(), scorer=match())
    outs = iter([ModelOutput.from_content("mockllm/model", x) for x in ["2", "4", "2", "x", "2", "4"]])
    inspect_eval(tiny(), model="mockllm/model", model_args={"custom_outputs": outs}, epochs=3,
                 log_dir=str(tmp_path), log_format="eval", display="none", max_connections=1)
    t = read_logs(tmp_path)
    assert len(t.trials) == 6 and t.warnings == []


SCRIPT = textwrap.dedent('''
    import os, re, sys, json, random
    import tiktoken
    sys.path.insert(0, {tests!r})
    import conftest  # applies the offline tokenizer patch if needed
    from inspect_ai import eval, Task, task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate

    @task
    def arith():
        return Task(dataset=[Sample(input=f"What is {{i}}+{{i}}?", target=str(2*i), id=f"t{{i}}") for i in range(4)],
                    solver=generate(), scorer=match())
    live = [0]
    def gen(messages, tools, tc, cfg):
        live[0] += 1
        i = int(re.search(r"What is (\\d+)\\+", messages[-1].text).group(1))
        return ModelOutput.from_content("mockllm/model", str(2*i) if random.random() < 0.5 else "no")
    mode = sys.argv[1]
    kw = dict(model_args={{"custom_outputs": gen}}) if mode == "record" else {{}}
    log = eval(arith(), model=f"passk/{{mode}}/mockllm/model", epochs=4, log_dir=sys.argv[2],
               log_format="json", display="none", **kw)[0]
    print(json.dumps({{"live": live[0], "status": log.status,
                      "scores": sorted((s.id, s.epoch, s.scores["match"].value) for s in log.samples)}}))
''')


def run(mode, logdir, cass, tests_dir):
    env = {**os.environ, "PASSK_CASSETTE_DIR": str(cass), "PASSK_REPORT": "1"}
    p = subprocess.run([sys.executable, "-c", SCRIPT.format(tests=str(tests_dir)), mode, str(logdir)],
                       env=env, capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_record_replay_provider_and_report_hook(tmp_path):
    from pathlib import Path
    tests_dir = Path(__file__).parent
    cass = tmp_path / "cass"
    rec = run("record", tmp_path / "rec", cass, tests_dir)
    assert rec["live"] == 16 and len(list(cass.glob("*.json"))) == 16  # 4 tasks x 4 epochs, none collapsed
    rep = run("replay", tmp_path / "rep", cass, tests_dir)
    assert rep["live"] == 0 and rep["status"] == "success"
    assert rep["scores"] == rec["scores"]  # byte-identical outcomes with no model at all
    out = list((tmp_path / "rec" / "passk").glob("*.json"))
    assert out, "PASSK_REPORT=1 should write a report next to the Inspect log"
    assert json.loads(out[0].read_text())["schema_version"] == 2
    assert list((tmp_path / "rec" / "passk").glob("*.html"))
