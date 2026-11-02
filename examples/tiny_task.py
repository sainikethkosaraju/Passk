"""Tiny Inspect task for the collect workflow (public, synthetic data only)."""
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate


@task
def tiny():
    return Task(
        dataset=[Sample(input=f"What is {i}+{i}? Answer with the number only.",
                        target=str(2 * i), id=f"t{i}") for i in range(1, 6)],
        solver=generate(),
        scorer=match(),
    )
