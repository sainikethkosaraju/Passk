"""Token usage -> USD using a dated, hash-recorded LiteLLM price snapshot.

*Shadow cost*: free-tier open-weight calls cost $0 in reality; shadow cost is what the same tokens
would cost at list price. Model names in logs rarely equal LiteLLM keys, so resolution is explicit
and recorded: exact -> provider-prefix stripped -> alias file -> median across providers that list
the base model. Long-context tier (`*_above_200k_tokens`) is applied per sample total (Inspect logs
aggregate usage per sample): an approximation, documented."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import statistics
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passk.logs import Trial, TrialTable

STALE_DAYS = 120
_PKG_PRICES = Path(__file__).parent / "_prices"
_REPO_PRICES = Path(__file__).resolve().parents[2] / "data" / "prices"
PRICE_FIELDS = ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost",
                "input_cost_per_token_above_200k_tokens", "output_cost_per_token_above_200k_tokens",
                "litellm_provider", "mode")
DEFAULT_INCLUDE = (r"gpt-|^o[1-9]|claude|gemini|llama|mistral|mixtral|deepseek|qwen|nemotron|gpt-oss|"
                   r"command-r|grok|kimi|glm")


def prices_dir() -> Path:
    return _PKG_PRICES if _PKG_PRICES.exists() else _REPO_PRICES


@dataclass
class Resolved:
    entry: dict[str, Any]
    how: str


@dataclass
class PriceBook:
    date: dt.date
    models: dict[str, dict]
    source: str
    aliases: dict[str, str]
    sha256: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceBook:
        if path is None:
            files = sorted(prices_dir().glob("prices_*.json"))
            if not files:
                raise FileNotFoundError("no price snapshot found")
            path = files[-1]
        raw = json.loads(Path(path).read_text())
        meta = raw.get("_meta", {})
        book = cls(dt.date.fromisoformat(meta["date"]),
                   {k: v for k, v in raw.items() if not k.startswith("_")},
                   meta.get("source", "unknown"), raw.get("_aliases", {}), meta.get("source_sha256", ""))
        age = (dt.date.today() - book.date).days
        if age > STALE_DAYS:
            warnings.warn(f"price snapshot {book.date} is {age} days old; costs may be stale",
                          stacklevel=2)
        return book

    def resolve(self, model: str) -> Resolved:
        target = self.aliases.get(model, model)
        r = self._resolve(target)
        if target != model:
            r = Resolved(r.entry, "alias" if r.how == "exact" else f"alias+{r.how}")
        return r

    def _resolve(self, name: str) -> Resolved:
        if name in self.models:
            return Resolved(self.models[name], "exact")
        parts = name.split("/")
        for i in range(1, len(parts)):
            cand = "/".join(parts[i:])
            if cand in self.models:
                return Resolved(self.models[cand], "stripped-prefix")
        base = parts[-1].lower()
        hits = [v for k, v in self.models.items()
                if k.lower().split("/")[-1] == base and v.get("input_cost_per_token", 0) > 0]
        if hits:
            med = {f: statistics.median(h[f] for h in hits if f in h)
                   for f in ("input_cost_per_token", "output_cost_per_token")}
            cache = [h["cache_read_input_token_cost"] for h in hits if "cache_read_input_token_cost" in h]
            if cache:
                med["cache_read_input_token_cost"] = statistics.median(cache)
            return Resolved(med, f"median-of-{len(hits)}-providers")
        raise KeyError(f"no price for model {name!r}")

    def lookup(self, model: str) -> dict:
        return self.resolve(model).entry


def price(usage: dict[str, int], model: str, book: PriceBook | None = None) -> float:
    """USD for a usage dict with input_tokens, output_tokens, cache_read_tokens."""
    book = book or PriceBook.load()
    p = book.lookup(model)
    inp, out = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    cached = min(int(usage.get("cache_read_tokens", 0)), inp)
    long_ctx = inp > 200_000
    in_rate = p.get("input_cost_per_token_above_200k_tokens", p["input_cost_per_token"]) \
        if long_ctx else p["input_cost_per_token"]
    out_rate = p.get("output_cost_per_token_above_200k_tokens", p["output_cost_per_token"]) \
        if long_ctx else p["output_cost_per_token"]
    cache_rate = p.get("cache_read_input_token_cost", in_rate)
    return (inp - cached) * in_rate + cached * cache_rate + out * out_rate


def trial_cost(t: Trial, book: PriceBook) -> float:
    return price({"input_tokens": t.input_tokens, "output_tokens": t.output_tokens,
                  "cache_read_tokens": t.cache_read_tokens}, t.model, book)


def cost_summary(table: TrialTable, book: PriceBook | None = None) -> dict:
    """Total shadow cost, cost per successful trial, and per-model resolution provenance."""
    from passk import stats
    book = book or PriceBook.load()
    total, missing, how = 0.0, set(), {}
    per_trial: list[tuple[float, bool]] = []
    for t in table.trials:
        try:
            c = trial_cost(t, book)
            how.setdefault(t.model, book.resolve(t.model).how)
        except KeyError:
            missing.add(t.model)
            continue
        total += c
        per_trial.append((c, t.success))
    successes = sum(ok for _, ok in per_trial)
    ci = None
    if successes and len(per_trial) >= 5:
        import random
        rng = random.Random(0)
        n = len(per_trial)
        vals = []
        for _ in range(500):
            s = [per_trial[rng.randrange(n)] for _ in range(n)]
            ok = sum(x[1] for x in s)
            if ok:
                vals.append(sum(x[0] for x in s) / ok)
        if vals:
            vals.sort()
            ci = [stats._quantile(vals, 0.025), stats._quantile(vals, 0.975)]
    return {"total_usd": total, "trials": len(table.trials), "successes": successes,
            "usd_per_success": (total / successes) if successes else None,
            "usd_per_success_ci": ci, "missing_prices": sorted(missing),
            "price_resolution": how, "price_snapshot": book.date.isoformat(),
            "price_source": book.source}


def pareto_front(points: list[tuple[str, float, float]]) -> list[str]:
    """points = (label, cost, reliability). Returns labels not dominated (lower cost AND higher
    reliability, at least one strict)."""
    keep = []
    for lab, c, r in points:
        if not any((c2 <= c and r2 >= r) and (c2 < c or r2 > r) for l2, c2, r2 in points if l2 != lab):
            keep.append(lab)
    return keep


def build_snapshot(source: str | Path, out_dir: str | Path, include: str = DEFAULT_INCLUDE,
                   today: dt.date | None = None, aliases: dict[str, str] | None = None) -> Path:
    """Create `prices_<date>.json` from a LiteLLM `model_prices_and_context_window.json`
    (URL, file:// URL or path), keeping chat models matching `include`. Records source + sha256."""
    src = str(source)
    if src.startswith(("http://", "https://", "file://")):
        with urllib.request.urlopen(src, timeout=60) as r:  # noqa: S310 - operator-supplied source
            raw = r.read()
    else:
        raw = Path(src).read_bytes()
    data = json.loads(raw)
    rx = re.compile(include, re.I)
    snap: dict[str, Any] = {}
    for k, v in sorted(data.items()):
        if not isinstance(v, dict) or v.get("mode") != "chat" or "input_cost_per_token" not in v:
            continue
        if rx.search(k):
            snap[k] = {f: v[f] for f in PRICE_FIELDS if f in v}
    if not snap:
        raise ValueError("no matching chat models in source")
    today = today or dt.date.today()
    meta = {"date": today.isoformat(), "source": src, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "include": include, "entries": len(snap)}
    out = Path(out_dir) / f"prices_{today.isoformat()}.json"
    out.write_text(json.dumps({"_meta": meta, "_aliases": aliases or {}, **snap}, indent=0,
                              sort_keys=True))
    return out
