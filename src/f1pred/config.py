"""Runtime settings, overridable with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    jolpica_base_url: str = os.environ.get("F1PRED_JOLPICA_URL", "https://api.jolpi.ca/ergast/f1")
    data_dir: Path = field(default_factory=lambda: _env_path("F1PRED_DATA_DIR", "data"))
    # Jolpica's published limits: 4 requests/second burst, 500 requests/hour sustained.
    requests_per_second: int = 4
    requests_per_hour: int = 500
    # Results for the current season can still change, so they are re-fetched after this.
    current_season_cache_hours: float = 6.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "http_cache.sqlite"

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"

    @property
    def model_dir(self) -> Path:
        """Where `f1pred evaluate` saves the models that `f1pred serve` loads."""
        return self.data_dir / "reports" / "model" / "models"
