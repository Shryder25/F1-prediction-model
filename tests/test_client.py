import httpx
import pytest
import respx

from f1pred.config import Settings
from f1pred.jolpica.cache import ResponseCache
from f1pred.jolpica.client import JolpicaClient, JolpicaError
from f1pred.jolpica.ratelimit import SlidingWindowLimiter
from tests.conftest import laps, mrdata, race, results

BASE = "https://api.test/ergast/f1"


def make_client(settings: Settings) -> JolpicaClient:
    return JolpicaClient(
        settings,
        cache=ResponseCache(":memory:"),
        limiter=SlidingWindowLimiter([(1000, 1.0)]),
    )


@respx.mock
async def test_results_parsed_and_typed(settings: Settings) -> None:
    respx.get(f"{BASE}/2024/1/results.json").respond(
        json=mrdata("RaceTable", "Races", [race(Results=results())], total=3)
    )
    async with make_client(settings) as client:
        got = await client.results(2024, 1)
    assert got is not None
    assert got.season == 2024
    assert [r.driver.code for r in got.results] == ["VER", "NOR", "LEC"]
    assert got.results[0].points == 25.0
    assert got.circuit.location.lat == pytest.approx(26.0325)


@respx.mock
async def test_race_split_across_pages_is_merged(settings: Settings) -> None:
    all_laps = laps(2)  # 2 laps x 3 drivers = 6 timing rows
    page1 = race(Laps=[all_laps[0], {"number": "2", "Timings": all_laps[1]["Timings"][:1]}])
    page2 = race(Laps=[{"number": "2", "Timings": all_laps[1]["Timings"][1:]}])
    route = respx.get(f"{BASE}/2024/1/laps.json")
    route.side_effect = [
        httpx.Response(200, json=mrdata("RaceTable", "Races", [page1], total=150)),
        httpx.Response(200, json=mrdata("RaceTable", "Races", [page2], total=150, offset=100)),
    ]
    async with make_client(settings) as client:
        got = await client.laps(2024, 1)
    assert got is not None
    assert [lap.number for lap in got.laps] == [1, 2]
    assert len(got.laps[1].timings) == 3
    assert route.call_count == 2


@respx.mock
async def test_responses_are_cached(settings: Settings) -> None:
    route = respx.get(f"{BASE}/2023/1/results.json").respond(
        json=mrdata("RaceTable", "Races", [race(2023, Results=results())], total=3)
    )
    async with make_client(settings) as client:
        await client.results(2023, 1)
        await client.results(2023, 1)
    assert route.call_count == 1


@respx.mock
async def test_retries_on_429(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("f1pred.jolpica.client.asyncio.sleep", no_sleep)
    route = respx.get(f"{BASE}/2024/1/results.json")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "1"}),
        httpx.Response(200, json=mrdata("RaceTable", "Races", [race(Results=results())], total=3)),
    ]
    async with make_client(settings) as client:
        got = await client.results(2024, 1)
    assert got is not None
    assert route.call_count == 2


@respx.mock
async def test_client_errors_are_not_retried(settings: Settings) -> None:
    route = respx.get(f"{BASE}/2024/1/results.json").respond(404)
    async with make_client(settings) as client:
        with pytest.raises(JolpicaError):
            await client.results(2024, 1)
    assert route.call_count == 1
