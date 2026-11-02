"""Generate REAL Inspect logs with the mockllm provider (zero cost).
tiktoken's vocab download is blocked in some sandboxes, so a whitespace tokenizer is patched in
here ONLY for log generation; everything else (task_run, scoring, log writing) is genuine Inspect."""
import sys

import tiktoken


class _Enc:
    def encode(self, text, **kw): return text.split()
    def decode(self, toks, **kw): return " ".join(toks)
tiktoken.get_encoding = lambda *a, **k: _Enc()
tiktoken.encoding_for_model = lambda *a, **k: _Enc()

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate

RATES = [1.0, 1.0, 0.9, 0.8, 0.8, 0.6, 0.5, 0.2, 0.0, 1.0]
N, EPOCHS = len(RATES), 5

@task
def arith():
    return Task(dataset=[Sample(input=f"What is {i}+{i}? Answer with the number only.", target=str(2*i), id=f"t{i}")
                         for i in range(N)], solver=generate(), scorer=match())

import hashlib
import re


def outputs(rates, seed):
    """Callable model: correctness is a deterministic function of (seed, task, call index), so the
    logs are reproducible regardless of the scheduling order Inspect chooses."""
    counts = {}
    def gen(messages, tools, tool_choice, config):
        prompt = messages[-1].text
        i = int(re.search(r"What is (\d+)\+", prompt).group(1))
        n = counts[i] = counts.get(i, 0) + 1
        h = int(hashlib.sha256(f"{seed}:{i}:{n}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        ok = h < rates[i]
        o = ModelOutput.from_content("mockllm/model", str(2*i) if ok else "I don't know")
        o.usage = ModelUsage(input_tokens=40+i, output_tokens=3 + (0 if ok else 40), total_tokens=43+i,
                             input_tokens_cache_read=8)
        return o
    return gen

out = sys.argv[1]
for name, rates, seed, fmt in [("baseline", RATES, 1, "json"), ("regressed", [max(0, r-0.4) for r in RATES], 2, "json"),
                               ("baseline", RATES, 3, "eval")]:
    m = get_model("mockllm/model", custom_outputs=outputs(rates, seed))
    eval(arith(), model=m, epochs=EPOCHS, log_dir=f"{out}/{name}-{fmt}", log_format=fmt, display="none",
         max_connections=1, log_samples=True)
print("done")
