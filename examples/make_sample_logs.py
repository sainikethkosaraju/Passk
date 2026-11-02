"""Generate synthetic Inspect-schema logs (seeded). Usage: python make_sample_logs.py OUTDIR

Field names mirror inspect-ai 0.3.287 sample logs (id, epoch, scores, model_usage, total_time,
error). These are SYNTHETIC: they were not produced by a live model.
"""
import json
import random
import sys
from pathlib import Path

RATES_A = [1.0, 1.0, 0.9, 0.8, 0.8, 0.6, 0.5, 0.2, 0.0, 1.0]


def make(path, rates, model="gpt-oss-120b", epochs=5, seed=1):
    rng = random.Random(seed)
    samples = []
    for i, r in enumerate(rates):
        for e in range(1, epochs + 1):
            ok = rng.random() < r
            samples.append({
                "id": f"task-{i}", "epoch": e, "input": "q", "target": "a",
                "scores": {"match": {"value": "C" if ok else "I"}},
                "model_usage": {model: {"input_tokens": 1000 + 50 * i, "output_tokens": 200,
                                        "input_tokens_cache_read": 400, "total_tokens": 1200}},
                "total_time": round(2.0 + rng.random() * 3, 3), "error": None,
            })
    log = {"version": 2, "status": "success", "eval": {"task": "tiny", "model": model},
           "samples": samples}
    Path(path).write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    make(out / "sample_a.json", RATES_A)
    make(out / "sample_b_regressed.json", [max(0.0, r - 0.4) for r in RATES_A], seed=2)
    make(out / "sample_a_rerun.json", RATES_A, seed=3)
