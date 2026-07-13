"""Canonical persistence for workflow research contracts."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.routing.contracts import (
    ActionOutcome,
    WorkflowAction,
    WorkflowRequest,
)
from small_models_society.schemas import StrictModel

ContractT = TypeVar("ContractT", bound=BaseModel)


class RoutingArtifact(StrictModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_contracts(
    path: Path,
    records: Sequence[BaseModel],
) -> RoutingArtifact:
    if not records:
        raise ValueError("routing artifact requires at least one row")
    lines = [canonical_json(record.model_dump(mode="json")) for record in records]
    content = ("\n".join(lines) + "\n").encode("utf-8")
    _write_atomic(path, content)
    return RoutingArtifact(
        path=path,
        sha256=sha256_bytes(content),
        row_count=len(records),
    )


def _load_contracts(
    path: Path,
    validator: Callable[[str], ContractT],
    description: str,
) -> list[ContractT]:
    records: list[ContractT] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(validator(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid {description} row at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"{description} artifact contains no rows: {path}")
    return records


def write_workflow_requests(
    path: Path,
    requests: Sequence[WorkflowRequest],
) -> RoutingArtifact:
    request_ids = [request.request_id for request in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("workflow requests contain duplicate request IDs")
    return _write_contracts(path, requests)


def load_workflow_requests(path: Path) -> list[WorkflowRequest]:
    requests = _load_contracts(
        path,
        WorkflowRequest.model_validate_json,
        "workflow request",
    )
    request_ids = [request.request_id for request in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError(f"workflow requests contain duplicate request IDs: {path}")
    return requests


def write_workflow_actions(
    path: Path,
    actions: Sequence[WorkflowAction],
) -> RoutingArtifact:
    action_ids = [action.action_id for action in actions]
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("workflow actions contain duplicate action IDs")
    return _write_contracts(path, actions)


def load_workflow_actions(path: Path) -> list[WorkflowAction]:
    actions = _load_contracts(
        path,
        WorkflowAction.model_validate_json,
        "workflow action",
    )
    action_ids = [action.action_id for action in actions]
    if len(set(action_ids)) != len(action_ids):
        raise ValueError(f"workflow actions contain duplicate action IDs: {path}")
    return actions


def write_action_outcomes(
    path: Path,
    outcomes: Sequence[ActionOutcome],
) -> RoutingArtifact:
    keys = [(outcome.request_id, outcome.action_id) for outcome in outcomes]
    if len(set(keys)) != len(keys):
        raise ValueError("action outcomes contain duplicate request/action pairs")
    return _write_contracts(path, outcomes)


def load_action_outcomes(path: Path) -> list[ActionOutcome]:
    outcomes = _load_contracts(
        path,
        ActionOutcome.model_validate_json,
        "action outcome",
    )
    keys = [(outcome.request_id, outcome.action_id) for outcome in outcomes]
    if len(set(keys)) != len(keys):
        raise ValueError(f"action outcomes contain duplicate request/action pairs: {path}")
    return outcomes
