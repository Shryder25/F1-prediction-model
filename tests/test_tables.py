import pytest

from f1pred.ingest.tables import lap_time_to_seconds, pit_stops_rows, results_rows
from f1pred.jolpica.models import Race
from tests.conftest import pit_stops, race, results


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("1:23.456", 83.456), ("23.456", 23.456), ("2:00.000", 120.0)],
)
def test_lap_time_to_seconds(text: str, seconds: float) -> None:
    assert lap_time_to_seconds(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", None])
def test_blank_lap_time_is_none(text: str | None) -> None:
    assert lap_time_to_seconds(text) is None


def test_results_rows_have_keys_for_joining() -> None:
    rows = results_rows(Race.model_validate(race(Results=results())))
    assert len(rows) == 3
    assert rows[0]["season"] == 2024
    assert rows[0]["driver_code"] == "VER"
    assert rows[0]["race_time_ms"] == 5_504_742


def test_pit_stop_duration_parsed() -> None:
    rows = pit_stops_rows(Race.model_validate(race(PitStops=pit_stops())))
    assert rows[0]["duration_s"] == pytest.approx(23.1)
