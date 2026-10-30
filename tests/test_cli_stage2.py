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


def test_pipeline_runs_the_steps_and_writes_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("F1PRED_DATA_DIR", str(tmp_path))
    assert main(["demo", "--seasons", "2019-2026"]) == 0
    assert main(["--demo", "pipeline", "--from", "check", "--quick"]) == 0
    out = capsys.readouterr().out
    assert "=== f1pred transform ===" in out
    assert "=== f1pred evaluate --quick ===" in out
    results = (tmp_path / "demo" / "reports" / "results.md").read_text()
    assert "Synthetic demo data" in results
    assert "Seasons loaded: 2019-2026." in results
    assert "pole converted to a win" in results
    assert "## Model report" in results
    assert (tmp_path / "demo" / "reports" / "model" / "models" / "lap.joblib").exists()


def test_pipeline_stops_at_the_first_failed_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import f1pred.cli as cli

    calls: list[list[str]] = []
    real_main = cli.main

    def fake_main(argv: list[str] | None = None) -> int:
        assert argv is not None
        if argv[0] == "pipeline":
            return real_main(argv)
        calls.append(argv)
        return 1 if argv[0] == "load" else 0

    monkeypatch.setenv("F1PRED_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "main", fake_main)
    assert cli.main(["pipeline", "--seasons", "2024"]) == 1
    assert [c[0] for c in calls] == ["ingest", "ingest-fastf1", "load"]
    assert calls[0] == ["ingest", "--seasons", "2024"]
    assert "resume with: f1pred pipeline --from load" in capsys.readouterr().out
