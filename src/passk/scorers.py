"""Inspect scorers used by imported evals (optional: needs inspect-ai)."""

from __future__ import annotations

from difflib import SequenceMatcher

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


@scorer(metrics=[accuracy(), stderr()])
def fuzzy_match(threshold: float = 0.8):
    """Correct if difflib similarity to any target >= threshold (approximation of text_similarity)."""
    async def score(state: TaskState, target: Target) -> Score:
        out = state.output.completion
        best = max((similarity(out, t) for t in target), default=0.0)
        return Score(value=CORRECT if best >= threshold else INCORRECT, answer=out,
                     explanation=f"best similarity {best:.2f} (threshold {threshold})")
    return score
