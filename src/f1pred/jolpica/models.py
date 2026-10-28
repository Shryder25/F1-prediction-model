"""Typed models for Jolpica/Ergast responses.

The API returns every number as a string. Pydantic converts them, and fails loudly
if the API ever changes shape.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)


class Location(_Model):
    lat: float
    long: float
    locality: str
    country: str


class Circuit(_Model):
    circuit_id: str = Field(alias="circuitId")
    name: str = Field(alias="circuitName")
    location: Location = Field(alias="Location")


class Driver(_Model):
    driver_id: str = Field(alias="driverId")
    permanent_number: int | None = Field(default=None, alias="permanentNumber")
    code: str | None = None
    given_name: str = Field(alias="givenName")
    family_name: str = Field(alias="familyName")
    date_of_birth: date | None = Field(default=None, alias="dateOfBirth")
    nationality: str | None = None


class Constructor(_Model):
    constructor_id: str = Field(alias="constructorId")
    name: str
    nationality: str | None = None


class RaceTime(_Model):
    millis: int | None = None
    time: str


class Result(_Model):
    number: int
    position: int
    position_text: str = Field(alias="positionText")
    points: float
    driver: Driver = Field(alias="Driver")
    constructor: Constructor = Field(alias="Constructor")
    grid: int
    laps: int
    status: str
    time: RaceTime | None = Field(default=None, alias="Time")


class QualifyingResult(_Model):
    number: int
    position: int
    driver: Driver = Field(alias="Driver")
    constructor: Constructor = Field(alias="Constructor")
    q1: str | None = Field(default=None, alias="Q1")
    q2: str | None = Field(default=None, alias="Q2")
    q3: str | None = Field(default=None, alias="Q3")


class PitStop(_Model):
    driver_id: str = Field(alias="driverId")
    lap: int
    stop: int
    time: str
    duration: str


class Timing(_Model):
    driver_id: str = Field(alias="driverId")
    position: int
    time: str


class Lap(_Model):
    number: int
    timings: list[Timing] = Field(alias="Timings")


class Race(_Model):
    season: int
    round: int
    name: str = Field(alias="raceName")
    circuit: Circuit = Field(alias="Circuit")
    date: date
    results: list[Result] = Field(default_factory=list, alias="Results")
    qualifying: list[QualifyingResult] = Field(default_factory=list, alias="QualifyingResults")
    pit_stops: list[PitStop] = Field(default_factory=list, alias="PitStops")
    laps: list[Lap] = Field(default_factory=list, alias="Laps")


class DriverStanding(_Model):
    position: int | None = None
    points: float
    wins: int
    driver: Driver = Field(alias="Driver")
    constructors: list[Constructor] = Field(alias="Constructors")


class ConstructorStanding(_Model):
    position: int | None = None
    points: float
    wins: int
    constructor: Constructor = Field(alias="Constructor")
