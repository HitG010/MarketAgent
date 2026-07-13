from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from small_models_society.data.prepare import canonical_json
from small_models_society.inference.contracts import ChatMessage
from small_models_society.routing.artifacts import write_workflow_requests
from small_models_society.routing.config import RoutingConfig, load_routing_config
from small_models_society.routing.contracts import (
    ActionAvailability,
    ActionOutcomeStatus,
    AvailabilityStatus,
    EnergyProvenance,
    OutputContract,
    RequestPolicyContext,
    SafetyAssessment,
    SafetyStatus,
    WorkflowAction,
    WorkflowRequest,
    create_workflow_request,
)
from small_models_society.routing.policy import (
    ActionRuntimeContext,
    evaluate_action_availability,
)
from small_models_society.routing.registry import ActionRegistry, build_action_registry
from small_models_society.routing.replay import (
    PricingCatalog,
    PricingEntry,
    ReplayCapture,
    ReplayCatalog,
    create_pricing_catalog,
    create_strong_replay_row,
    execute_strong_replay,
    import_replay_captures,
    inspect_replay_catalog,
    load_strong_replay_rows,
    load_verified_replay_catalog,
    replay_action_ids_for_request,
    replay_compatibility_fingerprint,
    verify_replay_rows,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "routing.yaml"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "routing"
ACTION_ID = "remote.strong-replay.reference.v1"


def _config() -> RoutingConfig:
    return load_routing_config(CONFIG_PATH)


def _registry() -> ActionRegistry:
    return build_action_registry(_config())


def _action() -> WorkflowAction:
    return _registry().actions[ACTION_ID].action


def _request(request_id: str = "routing-replay") -> WorkflowRequest:
    defaults = _config().policy_defaults
    return create_workflow_request(
        request_id=request_id,
        messages=(
            ChatMessage(role="system", content="Be precise."),
            ChatMessage(role="user", content=f"Question for {request_id}?"),
        ),
        output_contract=OutputContract.FREE_TEXT,
        policy=RequestPolicyContext(
            data_classification=defaults.data_classification,
            network_allowed=defaults.network_allowed,
            allowed_corpus_ids=defaults.allowed_corpus_ids,
            allowed_tool_ids=defaults.allowed_tool_ids,
            required_quality=defaults.required_quality,
            allow_unknown_output_safety=defaults.allow_unknown_output_safety,
        ),
    )


def _pricing() -> PricingCatalog:
    entry = PricingEntry(
        pricing_schedule_id="reference-usd-v1",
        provider_id="provider-neutral-replay",
        model_id="reference-strong-model",
        model_version="v1",
        prompt_usd_per_million_tokens=10,
        completion_usd_per_million_tokens=30,
    )
    return create_pricing_catalog({entry.pricing_schedule_id: entry})


def _capture(
    request: WorkflowRequest | None = None,
    action: WorkflowAction | None = None,
    **updates: object,
) -> ReplayCapture:
    resolved_request = request or _request()
    resolved_action = action or _action()
    values: dict[str, object] = {
        "request_id": resolved_request.request_id,
        "request_fingerprint": resolved_request.request_fingerprint,
        "action_id": resolved_action.action_id,
        "action_fingerprint": resolved_action.action_fingerprint,
        "provider_id": resolved_action.provider_id,
        "model_id": resolved_action.model_id,
        "model_version": resolved_action.model_version,
        "response": "Synthetic strong response",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "observed_latency_ms": 125.5,
        "provider_fee_usd": 0.0016,
        "pricing_schedule_id": resolved_action.pricing_schedule_id,
        "provider_safety": SafetyAssessment(
            status=SafetyStatus.SAFE,
            source="provider-safety-v1",
        ),
        "energy_provenance": EnergyProvenance.UNAVAILABLE,
        "captured_at_utc": datetime(2026, 1, 15, 12, tzinfo=UTC),
        "capture_source": "synthetic-fixture-v1",
        "capture_metadata": {"capture_id": "fixture-001", "region": "offline"},
    }
    values.update(updates)
    return ReplayCapture.model_validate(values)


def _write_captures(path: Path, captures: list[ReplayCapture]) -> None:
    path.write_text(
        "\n".join(canonical_json(capture.model_dump(mode="json")) for capture in captures) + "\n",
        encoding="utf-8",
    )


def _imported(
    tmp_path: Path,
    *,
    captures: list[ReplayCapture] | None = None,
    requests: list[WorkflowRequest] | None = None,
) -> tuple[ReplayCatalog, Path, Path]:
    resolved_requests = requests or [_request(), _request("routing-uncovered")]
    requests_path = tmp_path / "requests.jsonl"
    write_workflow_requests(requests_path, resolved_requests)
    captures_path = tmp_path / "captures.jsonl"
    _write_captures(captures_path, captures or [_capture(resolved_requests[0])])
    result = import_replay_captures(
        captures_path,
        requests_path,
        tmp_path / "rows.jsonl",
        _config(),
        _registry(),
        _pricing(),
    )
    catalog = load_verified_replay_catalog(
        result.rows_path,
        requests_path,
        _config(),
        _registry(),
        _pricing(),
    )
    return catalog, requests_path, result.manifest_path


def _available(action: WorkflowAction) -> ActionAvailability:
    return ActionAvailability(
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=AvailabilityStatus.AVAILABLE,
    )


def test_pricing_catalog_is_tamper_evident_and_computes_fee() -> None:
    catalog = _pricing()
    entry = catalog.entries["reference-usd-v1"]

    assert entry.provider_fee(100, 20) == pytest.approx(0.0016)
    assert catalog.catalog_fingerprint == catalog.calculated_fingerprint()

    value = catalog.model_dump(mode="json")
    value["entries"]["reference-usd-v1"]["completion_usd_per_million_tokens"] = 31
    with pytest.raises(ValidationError, match="catalog fingerprint"):
        PricingCatalog.model_validate(value)


def test_committed_synthetic_fixtures_import_and_verify(tmp_path: Path) -> None:
    from small_models_society.routing.replay import load_pricing_catalog

    pricing = load_pricing_catalog(FIXTURE_ROOT / "pricing.json")
    imported = import_replay_captures(
        FIXTURE_ROOT / "replay_captures.jsonl",
        FIXTURE_ROOT / "replay_requests.jsonl",
        tmp_path / "rows.jsonl",
        _config(),
        _registry(),
        pricing,
    )
    catalog = load_verified_replay_catalog(
        imported.rows_path,
        FIXTURE_ROOT / "replay_requests.jsonl",
        _config(),
        _registry(),
        pricing,
    )

    assert imported.rows_artifact.row_count == 1
    assert catalog.rows[0].response == "Paris"
    assert catalog.rows[0].row_sha256 == catalog.rows[0].calculated_sha256()


def test_import_is_canonical_deterministic_and_reports_partial_coverage(
    tmp_path: Path,
) -> None:
    requests = [_request(), _request("routing-uncovered")]
    requests_path = tmp_path / "requests.jsonl"
    write_workflow_requests(requests_path, requests)
    captures_path = tmp_path / "captures.jsonl"
    _write_captures(captures_path, [_capture(requests[0])])
    first = import_replay_captures(
        captures_path,
        requests_path,
        tmp_path / "first" / "rows.jsonl",
        _config(),
        _registry(),
        _pricing(),
    )
    second = import_replay_captures(
        captures_path,
        requests_path,
        tmp_path / "second" / "rows.jsonl",
        _config(),
        _registry(),
        _pricing(),
    )

    assert first.rows_path.read_bytes() == second.rows_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.rows_artifact.row_count == 1
    assert first.manifest.rows_sha256 == first.rows_artifact.sha256
    catalog = load_verified_replay_catalog(
        first.rows_path,
        requests_path,
        _config(),
        _registry(),
        _pricing(),
    )
    inspection = inspect_replay_catalog(catalog, requests)
    assert inspection.row_count == 1
    assert inspection.request_count == 2
    assert inspection.covered_request_count == 1
    assert inspection.request_coverage == 0.5
    assert inspection.rows_by_action == {ACTION_ID: 1}
    assert inspection.safety_status_counts == {"safe": 1}
    assert inspection.provider_fee_usd == pytest.approx(0.0016)
    assert inspection.energy_known_count == 0
    assert inspection.energy_known_rate == 0


def test_replay_compatibility_survives_unrelated_lora_approval(tmp_path: Path) -> None:
    request = _request()
    requests_path = tmp_path / "requests.jsonl"
    write_workflow_requests(requests_path, [request])
    captures_path = tmp_path / "captures.jsonl"
    _write_captures(captures_path, [_capture(request)])
    base_config = _config()
    base_registry = build_action_registry(base_config)
    imported = import_replay_captures(
        captures_path,
        requests_path,
        tmp_path / "rows.jsonl",
        base_config,
        base_registry,
        _pricing(),
    )
    action_id = "local.qwen-lora-math.v1"
    configured = base_config.actions[action_id]
    approved_config = base_config.model_copy(
        update={
            "actions": {
                **base_config.actions,
                action_id: configured.model_copy(update={"approved": True}),
            }
        }
    )
    approved_registry = build_action_registry(approved_config)

    assert base_config.fingerprint() != approved_config.fingerprint()
    assert base_registry.registry_fingerprint != approved_registry.registry_fingerprint
    assert replay_compatibility_fingerprint(base_config, base_registry) == (
        replay_compatibility_fingerprint(approved_config, approved_registry)
    )
    catalog = load_verified_replay_catalog(
        imported.rows_path,
        requests_path,
        approved_config,
        approved_registry,
        _pricing(),
    )
    assert catalog.rows[0].request_id == request.request_id


def test_verified_row_executes_without_network_and_preserves_observed_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    catalog, _, _ = _imported(
        tmp_path,
        requests=[request],
        captures=[_capture(request)],
    )
    action = _action()

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("replay execution attempted a network call")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    outcome = execute_strong_replay(request, action, _available(action), catalog)

    assert outcome.status is ActionOutcomeStatus.COMPLETED
    assert outcome.response == "Synthetic strong response"
    assert outcome.safety.status is SafetyStatus.SAFE
    assert outcome.telemetry is not None
    assert outcome.telemetry.wall_latency_ms == 125.5
    assert outcome.telemetry.prompt_tokens == 100
    assert outcome.telemetry.completion_tokens == 20
    assert outcome.telemetry.provider_fee_usd == pytest.approx(0.0016)
    assert outcome.telemetry.compute_cost_usd is None
    assert outcome.telemetry.total_cost_usd is None
    assert outcome.telemetry.energy_provenance is EnergyProvenance.UNAVAILABLE
    assert outcome.metadata["replay_row_sha256"] == catalog.rows[0].row_sha256


def test_measured_replay_energy_and_blocked_provider_safety_are_preserved(
    tmp_path: Path,
) -> None:
    request = _request()
    safety = SafetyAssessment(
        status=SafetyStatus.BLOCKED,
        source="provider-safety-v1",
        rule_ids=("provider.policy.block-v1",),
    )
    capture = _capture(
        request,
        provider_safety=safety,
        energy_joules=42.5,
        energy_provenance=EnergyProvenance.REPLAY,
        energy_measurement_source="provider-meter-v1",
    )
    catalog, _, _ = _imported(tmp_path, requests=[request], captures=[capture])

    outcome = execute_strong_replay(
        request,
        _action(),
        _available(_action()),
        catalog,
    )

    assert outcome.safety.status is SafetyStatus.BLOCKED
    assert outcome.telemetry is not None
    assert outcome.telemetry.energy_joules == 42.5
    assert outcome.telemetry.energy_provenance is EnergyProvenance.REPLAY
    assert outcome.telemetry.energy_measurement_source == "provider-meter-v1"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"request_fingerprint": "a" * 64}, "request fingerprint"),
        ({"action_fingerprint": "b" * 64}, "action fingerprint"),
        ({"model_version": "different"}, "provider/model identity"),
        ({"pricing_schedule_id": "unknown-v1"}, "pricing schedule"),
        ({"provider_fee_usd": 0.5}, "provider fee"),
        ({"completion_tokens": 513}, "token budget"),
        (
            {
                "provider_safety": SafetyAssessment(
                    status=SafetyStatus.UNKNOWN,
                    source="not-verified",
                )
            },
            "verified provider safety",
        ),
    ],
)
def test_import_rejects_mismatched_capture_provenance(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    request = _request()
    requests_path = tmp_path / "requests.jsonl"
    write_workflow_requests(requests_path, [request])
    captures_path = tmp_path / "captures.jsonl"
    _write_captures(captures_path, [_capture(request, **updates)])

    with pytest.raises(ValueError, match=message):
        import_replay_captures(
            captures_path,
            requests_path,
            tmp_path / "rows.jsonl",
            _config(),
            _registry(),
            _pricing(),
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"nested": {"reference": "hidden"}},
        {"domain": "knowledge"},
        {"api_key": "do-not-store"},
        {"headers": {"authorization": "do-not-store"}},
    ],
)
def test_capture_contract_rejects_evaluator_and_credential_metadata(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="evaluator or credential"):
        _capture(capture_metadata=metadata)


@pytest.mark.parametrize(
    "updates",
    [
        {"prompt_tokens": -1},
        {"completion_tokens": -1},
        {"observed_latency_ms": -0.1},
        {"provider_fee_usd": -0.1},
        {"energy_joules": -0.1},
    ],
)
def test_capture_contract_rejects_negative_telemetry(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _capture(**updates)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "energy_joules": 1,
            "energy_provenance": EnergyProvenance.UNAVAILABLE,
        },
        {
            "energy_joules": 1,
            "energy_provenance": EnergyProvenance.REPLAY,
        },
        {"energy_provenance": EnergyProvenance.ESTIMATED},
    ],
)
def test_capture_contract_rejects_invalid_energy_provenance(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="energy"):
        _capture(**updates)


def test_import_rejects_duplicate_request_action_captures(tmp_path: Path) -> None:
    request = _request()
    requests_path = tmp_path / "requests.jsonl"
    write_workflow_requests(requests_path, [request])
    captures_path = tmp_path / "captures.jsonl"
    capture = _capture(request)
    _write_captures(captures_path, [capture, capture])

    with pytest.raises(ValueError, match="duplicate request/action"):
        import_replay_captures(
            captures_path,
            requests_path,
            tmp_path / "rows.jsonl",
            _config(),
            _registry(),
            _pricing(),
        )


def test_import_rejects_valid_catalog_without_action_pricing_version(
    tmp_path: Path,
) -> None:
    request = _request()
    requests_path = tmp_path / "requests.jsonl"
    write_workflow_requests(requests_path, [request])
    captures_path = tmp_path / "captures.jsonl"
    _write_captures(captures_path, [_capture(request)])
    other_entry = PricingEntry(
        pricing_schedule_id="other-usd-v1",
        provider_id="provider-neutral-replay",
        model_id="reference-strong-model",
        model_version="v1",
        prompt_usd_per_million_tokens=10,
        completion_usd_per_million_tokens=30,
    )
    pricing = create_pricing_catalog({other_entry.pricing_schedule_id: other_entry})

    with pytest.raises(ValueError, match="unknown replay pricing schedule"):
        import_replay_captures(
            captures_path,
            requests_path,
            tmp_path / "rows.jsonl",
            _config(),
            _registry(),
            pricing,
        )


def test_tampered_verified_row_is_rejected(tmp_path: Path) -> None:
    _, _, _ = _imported(tmp_path)
    rows_path = tmp_path / "rows.jsonl"
    value = json.loads(rows_path.read_text(encoding="utf-8"))
    value["response"] = "Tampered response"
    rows_path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid replay row"):
        load_strong_replay_rows(rows_path)


def test_tampered_import_manifest_is_rejected_at_catalog_load(tmp_path: Path) -> None:
    _, requests_path, manifest_path = _imported(tmp_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["row_count"] = 2
    manifest_path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid replay import manifest"):
        load_verified_replay_catalog(
            tmp_path / "rows.jsonl",
            requests_path,
            _config(),
            _registry(),
            _pricing(),
        )


def test_missing_request_row_is_unavailable_and_never_fabricated(tmp_path: Path) -> None:
    covered = _request()
    missing = _request("routing-missing")
    catalog, _, _ = _imported(
        tmp_path,
        requests=[covered, missing],
        captures=[_capture(covered)],
    )
    registered = _registry().actions[ACTION_ID]
    runtime = ActionRuntimeContext(
        local_model_ready=False,
        verified_adapter_ids=(),
        available_corpus_ids=(),
        replay_action_ids=replay_action_ids_for_request(catalog, missing.request_id),
        calculator_supported=False,
    )
    availability = evaluate_action_availability(missing, registered, runtime)

    assert availability.status is AvailabilityStatus.UNAVAILABLE
    assert availability.reason_code == "replay_row_missing"
    with pytest.raises(ValueError, match="must not be executed"):
        execute_strong_replay(missing, registered.action, availability, catalog)
    with pytest.raises(ValueError, match="never fabricates"):
        execute_strong_replay(
            missing,
            registered.action,
            _available(registered.action),
            catalog,
        )


def test_import_refuses_collision_without_explicit_overwrite(tmp_path: Path) -> None:
    request = _request()
    requests_path = tmp_path / "requests.jsonl"
    write_workflow_requests(requests_path, [request])
    captures_path = tmp_path / "captures.jsonl"
    _write_captures(captures_path, [_capture(request)])
    output_path = tmp_path / "rows.jsonl"
    first = import_replay_captures(
        captures_path,
        requests_path,
        output_path,
        _config(),
        _registry(),
        _pricing(),
    )

    with pytest.raises(FileExistsError, match="overwrite"):
        import_replay_captures(
            captures_path,
            requests_path,
            output_path,
            _config(),
            _registry(),
            _pricing(),
        )
    overwritten = import_replay_captures(
        captures_path,
        requests_path,
        output_path,
        _config(),
        _registry(),
        _pricing(),
        overwrite=True,
    )
    assert overwritten.rows_artifact.sha256 == first.rows_artifact.sha256


def test_verified_row_rejects_stale_pricing_fingerprint() -> None:
    capture = _capture()
    row = create_strong_replay_row(capture, "a" * 64)

    with pytest.raises(ValueError, match="pricing fingerprint"):
        verify_replay_rows(
            [row],
            [_request()],
            _registry(),
            _pricing(),
            _config(),
        )
