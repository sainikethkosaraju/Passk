"""Token usage to USD using a dated LiteLLM-format price snapshot.

Shadow cost: for free-tier open-weight models the real spend is $0; the snapshot price is
what the same tokens would cost at list price. Prices are per token. Long-context tier:
`*_above_200k_tokens` keys apply when a call's input exceeds 200k tokens; because Inspect
logs aggregate usage per sample, Passk applies the tier to the per-sample total (documented
approximation).
"""

from __future__ import annotations

import datetime as dt
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from passk.logs import Trial, TrialTable

STALE_DAYS = 120
_PKG_PRICES = Path(__file__).parent / "_prices"
_REPO_PRICES = Path(__file__).resolve().parents[2] / "data" / "prices"


def prices_dir() -> Path:
    return _PKG_PRICES if _PKG_PRICES.exists() else _REPO_PRICES


@dataclass
class PriceBook:
    date: dt.date
    models: dict[str, dict]
    source: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceBook:
        if path is None:
            files = sorted(prices_dir().glob("prices_*.json"))
            if not files:
                raise FileNotFoundError("no price snapshot found")
            path = files[-1]
        raw = json.loads(Path(path).read_text())
        meta = raw.get("_meta", {})
        book = cls(
            date=dt.date.fromisoformat(meta["date"]),
            models={k: v for k, v in raw.items() if not k.startswith("_")},
            source=meta.get("source", "unknown"),
        )
        age = (dt.date.today() - book.date).days
        if age > STALE_DAYS:
            warnings.warn(
                f"price snapshot {book.date} is {age} days old; costs may be stale",
                stacklevel=2,
            )
        return book

    def lookup(self, model: str) -> dict:
        for key in (model, model.split("/", 1)[-1]):
            if key in self.models:
                return self.models[key]
        raise KeyError(f"no price for model {model!r}")


def price(usage: dict[str, int], model: str, book: PriceBook | None = None) -> float:
    """USD for a usage dict with input_tokens, output_tokens, cache_read_tokens."""
    book = book or PriceBook.load()
    p = book.lookup(model)
    inp = int(usage.get("input_tokens", 0))
    out = int(usage.get("output_tokens", 0))
    cached = min(int(usage.get("cache_read_tokens", 0)), inp)
    long_ctx = inp > 200_000
    in_rate = p.get("input_cost_per_token_above_200k_tokens", p["input_cost_per_token"]) \
        if long_ctx else p["input_cost_per_token"]
    out_rate = p.get("output_cost_per_token_above_200k_tokens", p["output_cost_per_token"]) \
        if long_ctx else p["output_cost_per_token"]
    cache_rate = p.get("cache_read_input_token_cost", in_rate)
    return (inp - cached) * in_rate + cached * cache_rate + out * out_rate


def trial_cost(t: Trial, book: PriceBook) -> float:
    return price(
        {"input_tokens": t.input_tokens, "output_tokens": t.output_tokens,
         "cache_read_tokens": t.cache_read_tokens},
        t.model, book,
    )


def cost_summary(table: TrialTable, book: PriceBook | None = None) -> dict:
    """Total shadow cost and cost per successful task (cost of all trials / successes)."""
    book = book or PriceBook.load()
    total = 0.0
    missing: set[str] = set()
    for t in table.trials:
        try:
            total += trial_cost(t, book)
        except KeyError:
            missing.add(t.model)
    successes = sum(t.success for t in table.trials)
    return {
        "total_usd": total,
        "trials": len(table.trials),
        "successes": successes,
        "usd_per_success": (total / successes) if successes else None,
        "missing_prices": sorted(missing),
        "price_snapshot": book.date.isoformat(),
        "price_source": book.source,
    }
