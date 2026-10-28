"""The asyncio pipeline: source -> features -> models -> updates.

    source.laps()  ──►  RacePredictor.after_lap()  ──►  RaceUpdate  ──►  SSE clients

Replays and live races go through the same `run_race`, so the only difference
between them is the `RaceSource` behind it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Protocol

import pandas as pd

from f1pred.live.events import LapSnapshot, RaceInfo, RaceUpdate
from f1pred.live.predictor import Models, RacePredictor

log = logging.getLogger(__name__)


class RaceSource(Protocol):
    """Anything that can describe a race and then stream its laps in order."""

    async def race_info(self) -> RaceInfo: ...

    async def pre_race_features(self) -> pd.DataFrame | None: ...

    def laps(self) -> AsyncIterator[LapSnapshot]: ...


async def run_race(
    source: RaceSource, models: Models, after_lap: int = -1
) -> AsyncIterator[RaceUpdate]:
    """Yield the pre-race view (lap 0), then one update per completed lap.

    Updates up to `after_lap` are computed (the features need every lap) but
    not yielded, so a client that reconnects resumes where it stopped. The last
    update always has `is_final` set, even if the race was cut short.
    """
    info = await source.race_info()
    predictor = RacePredictor(models, info)
    if after_lap < 0:
        yield predictor.before_start(await source.pre_race_features())
    last: RaceUpdate | None = None
    async for snapshot in source.laps():
        # Model calls take milliseconds; a thread keeps the event loop free anyway.
        last = await asyncio.to_thread(predictor.after_lap, snapshot)
        if last.lap > after_lap:
            yield last
    if last is not None and not last.is_final:
        yield last.model_copy(update={"is_final": True})


class Broadcaster:
    """Run one race in the background and fan its updates out to many clients.

    Used for live races, where every viewer must see the same feed. New
    subscribers first get every update so far, so a late joiner sees the whole
    race chart.
    """

    def __init__(self, updates: AsyncIterator[RaceUpdate]) -> None:
        self._updates = updates
        self.history: list[RaceUpdate] = []
        self._queues: set[asyncio.Queue[RaceUpdate | None]] = set()
        self._task: asyncio.Task[None] | None = None
        self.done = False
        self.error: str | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _pump(self) -> None:
        try:
            async for update in self._updates:
                self.history.append(update)
                for queue in list(self._queues):
                    queue.put_nowait(update)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # the feed failed: tell clients, keep the server up
            log.exception("Live race feed failed")
            self.error = str(error)
        finally:
            self.done = True
            for queue in list(self._queues):
                queue.put_nowait(None)

    async def subscribe(self, after_lap: int = -1) -> AsyncIterator[RaceUpdate]:
        queue: asyncio.Queue[RaceUpdate | None] = asyncio.Queue()
        # Copy the history and register in one step (no await between), so no
        # update is missed or sent twice.
        backlog = list(self.history)
        finished = self.done  # if so, no more updates (or end marker) will be queued
        self._queues.add(queue)
        try:
            for old in backlog:
                if old.lap > after_lap:
                    yield old
            if finished:
                return
            while (update := await queue.get()) is not None:
                if update.lap > after_lap:
                    yield update
        finally:
            self._queues.discard(queue)
