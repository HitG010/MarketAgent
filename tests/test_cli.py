from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from small_models_society.cli import main
from small_models_society.evaluation import load_predictions

FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


def test_fixtures_oracle_command_writes_predictions(tmp_path: Path) -> None:
    output = tmp_path / "oracle.jsonl"

    exit_code = main(
        [
            "fixtures",
            "oracle",
            "--benchmark",
            str(FIXTURE_BENCHMARK),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert len(load_predictions(output)) == 4
    assert '"reference"' not in output.read_text(encoding="utf-8")


def test_doctor_reports_unavailable_docker_without_crashing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["doctor"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code in {0, 1}
    assert "docker_daemon" in report["checks"]
    assert "sandbox_image" in report["checks"]


def test_doctor_rejects_stale_image_after_failed_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("small_models_society.cli.docker_available", lambda: True)
    monkeypatch.setattr("small_models_society.cli.sandbox_image_available", lambda _image: True)

    def fail_build(_image: str) -> None:
        raise subprocess.CalledProcessError(1, ["docker", "build"])

    monkeypatch.setattr("small_models_society.cli.build_sandbox_image", fail_build)

    exit_code = main(["doctor", "--build-sandbox"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["checks"]["sandbox_image"]["ok"] is False
