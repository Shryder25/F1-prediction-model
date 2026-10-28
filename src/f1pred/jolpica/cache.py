"""SQLite cache for raw API responses, so each URL is fetched once."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " url TEXT PRIMARY KEY, body TEXT NOT NULL, fetched_at REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, url: str, max_age_seconds: float | None = None) -> dict[str, Any] | None:
        """Return the cached body, or None if missing or older than max_age_seconds."""
        row = self._conn.execute(
            "SELECT body, fetched_at FROM responses WHERE url = ?", (url,)
        ).fetchone()
        if row is None:
            return None
        body, fetched_at = row
        if max_age_seconds is not None and time.time() - fetched_at > max_age_seconds:
            return None
        result: dict[str, Any] = json.loads(body)
        return result

    def put(self, url: str, body: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (url, body, fetched_at) VALUES (?, ?, ?)",
            (url, json.dumps(body), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
