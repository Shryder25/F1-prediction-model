"""Async rate limiter that enforces several sliding windows at once."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable


class SlidingWindowLimiter:
    """Allows at most `limit` calls in any `period` seconds, for each (limit, period) pair."""

    def __init__(
        self,
        windows: list[tuple[int, float]],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._windows = windows
        self._history: list[deque[float]] = [deque() for _ in windows]
        self._clock = clock
        self._sleep = sleep
        self._lock = asyncio.Lock()

    def _wait_time(self, now: float) -> float:
        wait = 0.0
        for (limit, period), history in zip(self._windows, self._history, strict=True):
            while history and now - history[0] >= period:
                history.popleft()
            if len(history) >= limit:
                wait = max(wait, period - (now - history[0]))
        return wait

    async def acquire(self) -> None:
        async with self._lock:
            while (wait := self._wait_time(self._clock())) > 0:
                await self._sleep(wait)
            now = self._clock()
            for history in self._history:
                history.append(now)
