"""Model-based grader with an explicit Unknown verdict and calibration against labels.

`judge` is any callable prompt -> str; verdict parsing is strict: only 'pass' or 'fail'
(case-insensitive, first token) count, everything else is 'unknown'. Unknown is never
silently treated as pass.
"""

from __future__ import annotations

from collections.abc import Callable


def parse_verdict(text: str) -> str:
    t = (text or "").strip().lower().split()
    first = t[0].strip(".,:;!") if t else ""
    return first if first in {"pass", "fail"} else "unknown"


def model_grade(judge: Callable[[str], str], question: str, answer: str,
                rubric: str) -> str:
    prompt = (f"Rubric: {rubric}\nQuestion: {question}\nAnswer: {answer}\n"
              "Reply with exactly one word: pass, fail, or unknown.")
    return parse_verdict(judge(prompt))


def calibrate(judge: Callable[[str], str],
              labeled: list[dict[str, str]]) -> dict[str, float | int]:
    """labeled items: {question, answer, rubric, label in {pass, fail}}."""
    tp = tn = fp = fn = unk = 0
    for item in labeled:
        v = model_grade(judge, item["question"], item["answer"], item["rubric"])
        if v == "unknown":
            unk += 1
        elif v == "pass" and item["label"] == "pass":
            tp += 1
        elif v == "fail" and item["label"] == "fail":
            tn += 1
        elif v == "pass":
            fp += 1
        else:
            fn += 1
    n = len(labeled)
    decided = n - unk
    return {
        "n": n, "unknown": unk, "unknown_rate": unk / n if n else 0.0,
        "agreement": (tp + tn) / decided if decided else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
