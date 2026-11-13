"""Resources shared by the Python assets and the dbt assets."""

from pathlib import Path

from dagster import ConfigurableResource

from f1pred.config import Settings


class DataLake(ConfigurableResource):  # type: ignore[type-arg]
    """Where raw files, caches and the warehouse live."""

    data_dir: str = "data"
    # Where debriefs are written. Real ones are committed and published to GitHub Pages.
    debrief_dir: str = "docs/debriefs"

    @property
    def settings(self) -> Settings:
        return Settings(data_dir=Path(self.data_dir))
