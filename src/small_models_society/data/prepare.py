"""Deterministic balanced benchmark preparation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from small_models_society.data.config import BenchmarkConfig, DatasetSource
from small_models_society.data.loaders import SourceRow, load_source_rows, normalize_row
from small_models_society.schemas import BenchmarkExample, Domain, validate_example

RowLoader = Callable[[DatasetSource], Iterable[SourceRow]]


@dataclass(frozen=True)
class PreparedBenchmark:
    benchmark_path: Path
    manifest_path: Path
    sha256: str
    row_count: int


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sample_key(example: BenchmarkExample, seed: int) -> str:
    return sha256_bytes(f"{seed}:{example.domain.value}:{example.id}".encode())


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def prepare_benchmark(
    config: BenchmarkConfig,
    output_dir: Path | None = None,
    row_loader: RowLoader | None = None,
) -> PreparedBenchmark:
    """Normalize and hash-rank a balanced sample from every configured domain."""

    loader = row_loader or load_source_rows
    selected: list[BenchmarkExample] = []
    source_row_counts: dict[str, int] = {}

    for domain in Domain:
        source = config.sources[domain]
        normalized = [normalize_row(domain, row, index) for index, row in enumerate(loader(source))]
        if len(normalized) < config.sample_per_domain:
            raise ValueError(
                f"{domain.value} has {len(normalized)} rows; {config.sample_per_domain} requested"
            )
        if len({example.id for example in normalized}) != len(normalized):
            raise ValueError(f"{domain.value} source contains duplicate normalized IDs")
        source_row_counts[domain.value] = len(normalized)
        selected.extend(
            sorted(normalized, key=lambda example: _sample_key(example, config.seed))[
                : config.sample_per_domain
            ]
        )

    selected.sort(key=lambda example: (example.domain.value, example.id))
    lines = [canonical_json(example.model_dump(mode="json")) for example in selected]
    benchmark_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    benchmark_sha256 = sha256_bytes(benchmark_bytes)
    destination = output_dir or Path(config.output_dir)
    benchmark_path = destination / "benchmark.jsonl"
    manifest_path = destination / "manifest.json"
    _write_atomic(benchmark_path, benchmark_bytes)

    counts = Counter(example.domain.value for example in selected)
    manifest = {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "sample_per_domain": config.sample_per_domain,
        "benchmark_file": benchmark_path.name,
        "benchmark_sha256": benchmark_sha256,
        "row_count": len(selected),
        "rows_by_domain": dict(sorted(counts.items())),
        "sources": {
            domain.value: {
                **config.sources[domain].model_dump(mode="json"),
                "available_rows": source_row_counts[domain.value],
                "sampled_rows": counts[domain.value],
            }
            for domain in Domain
        },
    }
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    _write_atomic(manifest_path, manifest_bytes)
    return PreparedBenchmark(
        benchmark_path=benchmark_path,
        manifest_path=manifest_path,
        sha256=benchmark_sha256,
        row_count=len(selected),
    )


def load_benchmark(path: Path) -> list[BenchmarkExample]:
    """Read and validate a normalized benchmark JSONL file."""

    examples: list[BenchmarkExample] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(validate_example(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid benchmark row at {path}:{line_number}") from error
    if len({example.id for example in examples}) != len(examples):
        raise ValueError(f"benchmark contains duplicate example IDs: {path}")
    return examples
