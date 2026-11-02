"""Test configuration. Inspect's token counter downloads a vocabulary on first use; some sandboxes
block that, so when (and only when) the download fails, a whitespace tokenizer is patched in. This
affects token COUNTS only; scoring, logging and Passk's code paths are unchanged."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "examples" / "inspect_logs"


def _patch_tiktoken_if_offline():
    try:
        import tiktoken
    except ImportError:
        return
    try:
        tiktoken.get_encoding("o200k_base")
        return
    except Exception:
        pass

    class _E:
        def encode(self, t, **k): return t.split()
        def decode(self, t, **k): return " ".join(t)
    tiktoken.get_encoding = lambda *a, **k: _E()
    tiktoken.encoding_for_model = lambda *a, **k: _E()


_patch_tiktoken_if_offline()


@pytest.fixture(scope="session")
def logs():
    return {"base": LOGS / "baseline-json", "reg": LOGS / "regressed-json", "eval": LOGS / "baseline-eval"}
