"""Token bucket and daily call budget for the scheduled collector (free-tier safe)."""

from __future__ import annotations

import time
from collections.abc import Callable


class BudgetExhausted(RuntimeError):
    pass


class TokenBucket:
    def __init__(self, rate_per_min: float, burst: int = 1,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.rate = rate_per_min / 60.0
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.clock, self.sleep = clock, sleep
        self.last = clock()

    def acquire(self) -> None:
        while True:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            self.sleep((1 - self.tokens) / self.rate)


class DailyBudget:
    def __init__(self, limit: int) -> None:
        self.limit, self.used = limit, 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise BudgetExhausted(f"daily budget of {self.limit} calls exhausted; resume tomorrow")
        self.used += 1
