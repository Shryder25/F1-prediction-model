"""Generate a small synthetic dataset shaped exactly like the real raw layer.

The demo lets anyone run the dbt models, tests and dashboard in a minute, with no
API access. It is also the fixture for the end-to-end tests: the generator knows
the true tyre degradation and pit crew speed it used, so the tests can check that
the warehouse models recover them.

Everything here is made up. The dashboard shows a banner when it reads demo data.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from f1pred import warehouse
from f1pred.ingest.pipeline import write_partition

DEMO_MARKER_TABLE = "_demo_data"
SEASONS = (2023, 2024)
POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)

# Seconds lost per lap of fuel on board. Matches the dbt var, so the fuel
# correction in the models removes it exactly.
FUEL_S_PER_LAP = 0.05
# True tyre degradation in seconds per lap of tyre age, before the circuit multiplier.
COMPOUND_DEG = {"SOFT": 0.09, "MEDIUM": 0.06, "HARD": 0.035}


@dataclass(frozen=True)
class Circuit:
    circuit_id: str
    name: str
    locality: str
    country: str
    lat: float
    lon: float
    laps: int
    base_lap_s: float
    pit_lane_s: float
    # Seconds of lap-1 gap per grid slot. Higher means the grid order holds.
    grid_gap_s: float
    deg_multiplier: float


@dataclass(frozen=True)
class Team:
    constructor_id: str
    pace_s: float  # lap time offset from the fastest car
    crew_delta_s: float  # pit crew speed relative to an average crew
    slow_stop_chance: float


@dataclass(frozen=True)
class DemoDriver:
    driver_id: str
    code: str
    number: int
    given: str
    family: str
    born: str
    team: str
    skill_s: float


CIRCUITS = (
    Circuit("bahrain", "Bahrain International Circuit", "Sakhir", "Bahrain",
            26.0325, 50.5106, 57, 95.0, 23.0, 0.25, 1.4),
    Circuit("monaco", "Circuit de Monaco", "Monte-Carlo", "Monaco",
            43.7347, 7.42056, 78, 74.0, 21.0, 2.5, 0.5),
    Circuit("monza", "Autodromo Nazionale di Monza", "Monza", "Italy",
            45.6156, 9.28111, 53, 82.0, 25.0, 0.4, 0.8),
    Circuit("silverstone", "Silverstone Circuit", "Silverstone", "UK",
            52.0786, -1.01694, 52, 89.0, 20.0, 0.35, 1.2),
)  # fmt: skip

TEAMS = {
    t.constructor_id: t
    for t in (
        Team("red_bull", 0.0, -0.6, 0.03),
        Team("mclaren", 0.15, -0.4, 0.05),
        Team("ferrari", 0.25, 0.3, 0.12),
        Team("mercedes", 0.35, 0.0, 0.06),
        Team("williams", 1.1, 0.5, 0.10),
    )
}

DRIVERS = (
    DemoDriver("max_verstappen", "VER", 1, "Max", "Verstappen", "1997-09-30", "red_bull", 0.0),
    DemoDriver("perez", "PER", 11, "Sergio", "Perez", "1990-01-26", "red_bull", 0.35),
    DemoDriver("norris", "NOR", 4, "Lando", "Norris", "1999-11-13", "mclaren", 0.05),
    DemoDriver("piastri", "PIA", 81, "Oscar", "Piastri", "2001-04-06", "mclaren", 0.12),
    DemoDriver("leclerc", "LEC", 16, "Charles", "Leclerc", "1997-10-16", "ferrari", 0.05),
    DemoDriver("sainz", "SAI", 55, "Carlos", "Sainz", "1994-09-01", "ferrari", 0.12),
    DemoDriver("hamilton", "HAM", 44, "Lewis", "Hamilton", "1985-01-07", "mercedes", 0.08),
    DemoDriver("russell", "RUS", 63, "George", "Russell", "1998-02-15", "mercedes", 0.08),
    DemoDriver("albon", "ALB", 23, "Alexander", "Albon", "1996-03-23", "williams", 0.1),
    DemoDriver("sargeant", "SAR", 2, "Logan", "Sargeant", "2000-12-31", "williams", 0.45),
)


@dataclass
class _Car:
    driver: DemoDriver
    grid: int
    pace_s: float
    retire_after: int | None
    stops: list[tuple[int, str]]  # (lap of pit entry, compound fitted)
    start_compound: str
    elapsed_s: float = 0.0
    last_lap_s: float = 0.0
    laps_done: int = 0


def _strategy(rng: random.Random, laps: int) -> tuple[str, list[tuple[int, str]]]:
    if rng.random() < 0.5:
        first = rng.choice(("MEDIUM", "SOFT"))
        return first, [(int(laps * rng.uniform(0.35, 0.5)), "HARD")]
    first_stop = int(laps * rng.uniform(0.25, 0.33))
    second_stop = int(laps * rng.uniform(0.6, 0.7))
    return "SOFT", [(first_stop, "MEDIUM"), (second_stop, "SOFT")]


def _race(rng: random.Random, season: int, round_: int, circuit: Circuit) -> dict[str, list[Any]]:
    """Simulate one race weekend. Returns rows per raw table."""
    out: dict[str, list[Any]] = {k: [] for k in ("qualifying", "results", "laps", "pit_stops")}
    out["fastf1_laps"] = []

    # Qualifying: one lap, ordered by pace.
    quali = sorted(
        (
            (circuit.base_lap_s - 2 + TEAMS[d.team].pace_s + d.skill_s + rng.gauss(0, 0.15), d)
            for d in DRIVERS
        ),
        key=lambda pair: pair[0],
    )
    cars: list[_Car] = []
    for pos, (lap_s, d) in enumerate(quali, start=1):
        out["qualifying"].append(
            {
                "driver_id": d.driver_id,
                "constructor_id": d.team,
                "position": pos,
                "q1_s": round(lap_s + 0.6, 3),
                "q2_s": round(lap_s + 0.3, 3) if pos <= 8 else None,
                "q3_s": round(lap_s, 3) if pos <= 5 else None,
            }
        )
        start, stops = _strategy(rng, circuit.laps)
        retire = rng.randint(5, circuit.laps - 1) if rng.random() < 0.06 else None
        pace = TEAMS[d.team].pace_s + d.skill_s
        cars.append(_Car(d, pos, pace, retire, stops, start))
    # One grid penalty per season sends a car to the pit lane.
    pit_lane_start = cars[rng.randrange(len(cars))] if round_ == 2 else None
    if pit_lane_start is not None:
        # The rest of the grid closes up, as it does in real results.
        for slot, car in enumerate((c for c in cars if c is not pit_lane_start), start=1):
            car.grid = slot
        pit_lane_start.grid = len(cars)

    safety_car = range(20, 24) if round_ == 3 else range(0)
    for lap in range(1, circuit.laps + 1):
        running = [c for c in cars if c.retire_after is None or lap <= c.retire_after]
        for car in running:
            stint = 1 + sum(1 for stop_lap, _ in car.stops if stop_lap < lap)
            compound = car.start_compound if stint == 1 else car.stops[stint - 2][1]
            stint_start = 1 if stint == 1 else car.stops[stint - 2][0] + 1
            tyre_life = lap - stint_start + 1
            pit_in = any(stop_lap == lap for stop_lap, _ in car.stops)
            pit_out = stint > 1 and lap == stint_start
            lap_s = (
                circuit.base_lap_s
                + car.pace_s
                + COMPOUND_DEG[compound] * circuit.deg_multiplier * tyre_life
                + FUEL_S_PER_LAP * (circuit.laps - lap)
                + rng.gauss(0, 0.12)
            )
            if lap == 1:
                lap_s += 4 + car.grid * circuit.grid_gap_s
                if car is pit_lane_start:
                    lap_s += 15
            if lap in safety_car:
                lap_s += 30
            if pit_out:
                lap_s += circuit.pit_lane_s - 2
            car.elapsed_s += lap_s
            car.last_lap_s = lap_s
            car.laps_done = lap
            out["fastf1_laps"].append(
                {
                    "driver_code": car.driver.code,
                    "driver_number": car.driver.number,
                    "team": car.driver.team,
                    "lap": lap,
                    "lap_time_s": round(lap_s, 3),
                    "stint": stint,
                    "compound": compound,
                    "tyre_life": tyre_life,
                    "fresh_tyre": True,
                    "pit_in": pit_in,
                    "pit_out": pit_out,
                    "track_status": "4" if lap in safety_car else "1",
                    "is_accurate": not (pit_in or pit_out or lap == 1 or lap in safety_car),
                    "safety_car": lap in safety_car,
                    "virtual_safety_car": False,
                    "red_flag": False,
                }
            )
            if pit_in:
                team = TEAMS[car.driver.team]
                duration = circuit.pit_lane_s + team.crew_delta_s + abs(rng.gauss(0, 0.3))
                if rng.random() < team.slow_stop_chance:
                    duration += rng.uniform(3, 8)
                out["pit_stops"].append(
                    {
                        "driver_id": car.driver.driver_id,
                        "stop": stint,
                        "lap": lap,
                        "duration_s": round(duration, 3),
                    }
                )
        order = sorted(running, key=lambda c: c.elapsed_s)
        for pos, car in enumerate(order, start=1):
            out["laps"].append(
                {
                    "lap": lap,
                    "driver_id": car.driver.driver_id,
                    "position": pos,
                    "lap_time_s": round(car.last_lap_s, 3),
                }
            )

    finishers = sorted((c for c in cars if c.retire_after is None), key=lambda c: c.elapsed_s)
    retired = sorted((c for c in cars if c.retire_after is not None), key=lambda c: -c.laps_done)
    for pos, car in enumerate(finishers + retired, start=1):
        classified = car.retire_after is None
        d = car.driver
        out["results"].append(
            {
                "driver_id": d.driver_id,
                "driver_code": d.code,
                "driver_number": d.number,
                "permanent_number": d.number,
                "given_name": d.given,
                "family_name": d.family,
                "date_of_birth": d.born,
                "constructor_id": d.team,
                "grid": 0 if car is pit_lane_start else car.grid,
                "position": pos,
                "position_text": str(pos) if classified else "R",
                "points": float(POINTS[pos - 1]) if classified and pos <= 10 else 0.0,
                "laps": car.laps_done,
                "status": "Finished" if classified else rng.choice(("Engine", "Collision")),
                "race_time_ms": int(car.elapsed_s * 1000) if classified else None,
            }
        )
    return out


def generate(raw_dir: Path, seed: int = 7, seasons: Sequence[int] = SEASONS) -> int:
    """Write a synthetic raw layer, two seasons by default. Returns rows written.

    The model tests pass more seasons, so there is enough history for a
    train / validation / test split by season.
    """
    rng = random.Random(seed)
    written = 0
    circuits = [
        {
            "circuit_id": c.circuit_id,
            "circuit_name": c.name,
            "locality": c.locality,
            "country": c.country,
            "lat": c.lat,
            "lon": c.lon,
        }
        for c in CIRCUITS
    ]
    path = raw_dir / "circuits" / "circuits.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(circuits).to_parquet(path, index=False)

    for season in seasons:
        driver_points: dict[str, float] = {d.driver_id: 0.0 for d in DRIVERS}
        driver_wins: dict[str, int] = {d.driver_id: 0 for d in DRIVERS}
        for round_, circuit in enumerate(CIRCUITS, start=1):
            race_date = f"{season}-{3 + 2 * round_:02d}-15"
            race_row = {
                "race_name": f"{circuit.country} Grand Prix",
                "circuit_id": circuit.circuit_id,
                "race_date": race_date,
            }
            write_partition(raw_dir, "races", [race_row], season, round_)
            tables = _race(rng, season, round_, circuit)
            for table, rows in tables.items():
                if write_partition(raw_dir, table, rows, season, round_):
                    written += len(rows)
            for r in tables["results"]:
                driver_points[r["driver_id"]] += r["points"]
                driver_wins[r["driver_id"]] += r["position_text"] == "1"
            written += _write_standings(raw_dir, season, round_, driver_points, driver_wins)
    return written


def _write_standings(
    raw_dir: Path,
    season: int,
    round_: int,
    points: dict[str, float],
    wins: dict[str, int],
) -> int:
    by_driver = sorted(points, key=lambda d: (-points[d], -wins[d]))
    team_of = {d.driver_id: d.team for d in DRIVERS}
    team_points: dict[str, float] = {}
    team_wins: dict[str, int] = {}
    for driver_id, pts in points.items():
        team = team_of[driver_id]
        team_points[team] = team_points.get(team, 0.0) + pts
        team_wins[team] = team_wins.get(team, 0) + wins[driver_id]
    by_team = sorted(team_points, key=lambda t: (-team_points[t], -team_wins[t]))
    drivers = [
        {"driver_id": d, "position": i, "points": points[d], "wins": wins[d]}
        for i, d in enumerate(by_driver, start=1)
    ]
    teams = [
        {"constructor_id": t, "position": i, "points": team_points[t], "wins": team_wins[t]}
        for i, t in enumerate(by_team, start=1)
    ]
    write_partition(raw_dir, "driver_standings", drivers, season, round_)
    write_partition(raw_dir, "constructor_standings", teams, season, round_)
    return len(drivers) + len(teams)


def build(data_dir: Path, seed: int = 7, seasons: Sequence[int] = SEASONS) -> dict[str, int]:
    """Generate the demo raw layer and load it into `<data_dir>/warehouse.duckdb`."""
    raw_dir = data_dir / "raw"
    generate(raw_dir, seed, seasons)
    warehouse_path = data_dir / "warehouse.duckdb"
    counts = warehouse.load(raw_dir, warehouse_path)
    with duckdb.connect(str(warehouse_path)) as con:
        con.execute(f"CREATE OR REPLACE TABLE {DEMO_MARKER_TABLE} AS SELECT {seed} AS seed")
    return counts
