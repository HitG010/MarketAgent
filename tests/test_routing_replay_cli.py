from __future__ import annotations

import json
from pathlib import Path

import pytest

from small_models_society.cli import main

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "routing.yaml"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "routing"


def _import_arguments(output: Path) -> list[str]:
    return [
        "routing",
        "replay-import",
        "--config",
        str(CONFIG_PATH),
        "--requests",
        str(FIXTURE_ROOT / "replay_requests.jsonl"),
        "--captures",
        str(FIXTURE_ROOT / "replay_captures.jsonl"),
        "--pricing",
        str(FIXTURE_ROOT / "pricing.json"),
        "--output",
        str(output),
    ]


def test_cli_imports_then_inspects_verified_replay_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_path = tmp_path / "rows.jsonl"

    assert main(_import_arguments(rows_path)) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["row_count"] == 1
    assert imported["rows"] == str(rows_path)
    assert len(imported["rows_sha256"]) == 64
    assert Path(imported["manifest"]).is_file()

    assert (
        main(
            [
                "routing",
                "replay-inspect",
                "--config",
                str(CONFIG_PATH),
                "--requests",
                str(FIXTURE_ROOT / "replay_requests.jsonl"),
                "--pricing",
                str(FIXTURE_ROOT / "pricing.json"),
                "--rows",
                str(rows_path),
            ]
        )
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["row_count"] == 1
    assert inspected["request_count"] == 1
    assert inspected["covered_request_count"] == 1
    assert inspected["request_coverage"] == 1
    assert inspected["provider_fee_usd"] == pytest.approx(0.0016)
    assert inspected["energy_known_rate"] == 0
    assert inspected["safety_status_counts"] == {"safe": 1}


def test_cli_refuses_replay_output_collision_without_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    arguments = _import_arguments(rows_path)

    assert main(arguments) == 0
    capsys.readouterr()
    assert main(arguments) == 1

    assert "use overwrite explicitly" in capsys.readouterr().err
