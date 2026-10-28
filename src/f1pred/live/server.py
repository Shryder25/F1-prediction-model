"""FastAPI service: race replays and a live race as Server-Sent Events, plus the HUD.

    GET /                                  the HUD (one HTML page, no build step)
    GET /api/health                        models and warehouse status
    GET /api/races                         races that can be replayed
    GET /api/races/{race_id}/stream        one replay per viewer, ?interval=seconds per lap
    GET /api/live/stream                   the live race, shared by every viewer

Each SSE message is `event: update`, `id: <lap>`, `data: <RaceUpdate JSON>`. A
browser that reconnects sends `Last-Event-ID`, and the stream resumes after that
lap instead of starting over.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from f1pred import insights
from f1pred.live.events import RaceUpdate
from f1pred.live.pipeline import Broadcaster, RaceSource, run_race
from f1pred.live.predictor import Models
from f1pred.live.replay import WarehouseReplay, list_races

log = logging.getLogger(__name__)

HUD = Path(__file__).parent / "static" / "hud.html"
# Comment lines keep idle connections open through proxies between live laps.
HEARTBEAT_S = 15.0

# Builds the live race source. None: this server has no live race.
LiveFactory = Callable[[], RaceSource]


def sse(update: RaceUpdate) -> str:
    return f"event: update\nid: {update.lap}\ndata: {update.model_dump_json()}\n\n"


async def _with_heartbeat(
    updates: AsyncIterator[RaceUpdate], every_s: float = HEARTBEAT_S
) -> AsyncIterator[str]:
    """SSE text for each update, with a comment line whenever the feed is quiet."""
    iterator = aiter(updates)
    pending: asyncio.Task[RaceUpdate] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=every_s)
            if not done:
                yield ": keep-alive\n\n"
                continue
            try:
                update = pending.result()
            except StopAsyncIteration:
                break
            pending = None
            yield sse(update)
        yield "event: end\ndata: {}\n\n"
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending


def _last_lap(last_event_id: str | None) -> int:
    try:
        return int(last_event_id) if last_event_id else -1
    except ValueError:
        return -1


def create_app(
    warehouse_path: Path,
    model_dir: Path,
    live: LiveFactory | None = None,
    models: Models | None = None,
) -> FastAPI:
    state: dict[str, Any] = {"models": models, "model_error": None, "live": None, "demo": False}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if warehouse_path.exists():
            state["demo"] = await asyncio.to_thread(_is_demo, warehouse_path)
        if state["models"] is None:
            try:
                state["models"] = await asyncio.to_thread(Models.load, model_dir)
            except FileNotFoundError as error:
                state["model_error"] = str(error)
                log.error("%s", error)
        broadcaster = None
        if live is not None and state["models"] is not None:
            broadcaster = Broadcaster(run_race(live(), state["models"]))
            broadcaster.start()
            state["live"] = broadcaster
        yield
        if broadcaster is not None:
            await broadcaster.stop()

    app = FastAPI(title="F1 race predictor", version="0.4.0", lifespan=lifespan)

    def _models() -> Models:
        if state["models"] is None:
            raise HTTPException(503, state["model_error"] or "Models are not loaded.")
        loaded: Models = state["models"]
        return loaded

    @app.get("/", include_in_schema=False)
    async def hud() -> FileResponse:
        return FileResponse(HUD, media_type="text/html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        broadcaster: Broadcaster | None = state["live"]
        return {
            "status": "ok" if state["models"] is not None else "degraded",
            "models_loaded": state["models"] is not None,
            "model_error": state["model_error"],
            "warehouse": warehouse_path.exists(),
            "demo_data": state["demo"],
            "live": None
            if broadcaster is None
            else {"laps": len(broadcaster.history), "done": broadcaster.done,
                  "error": broadcaster.error},
        }  # fmt: skip

    @app.get("/api/races")
    async def races() -> list[dict[str, Any]]:
        if not warehouse_path.exists():
            return []
        return [asdict(r) for r in await asyncio.to_thread(list_races, warehouse_path)]

    @app.get("/api/races/{race_id}/stream")
    async def replay(
        race_id: int,
        interval: Annotated[float, Query(ge=0.0, le=30.0)] = 1.0,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        models = _models()
        after = _last_lap(last_event_id)
        source = WarehouseReplay(warehouse_path, race_id, interval, fast_forward_to=max(after, 0))
        try:
            await source.race_info()
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        updates = run_race(source, models, after_lap=after)
        return _stream(updates)

    @app.get("/api/live/stream")
    async def live_stream(
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        broadcaster: Broadcaster | None = state["live"]
        if broadcaster is None:
            raise HTTPException(404, "No live race is configured on this server.")
        return _stream(broadcaster.subscribe(after_lap=_last_lap(last_event_id)))

    return app


def _is_demo(warehouse_path: Path) -> bool:
    with insights.connect(warehouse_path) as con:
        return insights.is_demo(con)


def _stream(updates: AsyncIterator[RaceUpdate]) -> StreamingResponse:
    return StreamingResponse(
        _with_heartbeat(updates),
        media_type="text/event-stream",
        # No caching, and tell nginx-style proxies not to buffer the stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
