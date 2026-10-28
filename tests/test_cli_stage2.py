from __future__ import annotations

from pathlib import Path

import pytest

from f1pred.cli import main


def test_demo_to_findings_from_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("F1PRED_DATA_DIR", str(tmp_path))
    assert main(["demo"]) == 0
    assert (tmp_path / "demo" / "warehouse.duckdb").exists()
    assert main(["--demo", "check"]) == 0
    assert main(["--demo", "transform", "--select", "+mart_grid_conversion"]) == 0
    assert main(["--demo", "transform"]) == 0
    capsys.readouterr()
    assert main(["--demo", "findings"]) == 0
    out = capsys.readouterr().out
    assert "demo data" in out
    assert "pole converted to a win" in out
