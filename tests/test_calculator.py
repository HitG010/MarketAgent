from __future__ import annotations

from pathlib import Path

import pytest

from small_models_society.inference.contracts import ChatMessage
from small_models_society.routing.config import (
    CalculatorOperator,
    RoutingConfig,
    load_routing_config,
)
from small_models_society.routing.contracts import (
    ActionAvailability,
    ActionOutcomeStatus,
    AvailabilityStatus,
    EnergyProvenance,
    OutputContract,
    RequestPolicyContext,
    SafetyStatus,
    WorkflowAction,
    WorkflowRequest,
    create_workflow_request,
)
from small_models_society.routing.policy import (
    RULE_CALCULATOR_INPUT,
    ActionRuntimeContext,
    evaluate_action_availability,
)
from small_models_society.routing.registry import build_action_registry
from small_models_society.tools.calculator import (
    CalculatorReason,
    CalculatorStatus,
    calculator_config_fingerprint,
    calculator_supported,
    evaluate_calculator_expression,
    evaluate_calculator_request,
    evaluate_calculator_suite,
    execute_calculator,
    measure_calculator_coverage,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "routing.yaml"


def _config() -> RoutingConfig:
    return load_routing_config(CONFIG_PATH)


def _request(
    expression: str,
    request_id: str = "calculator-request",
    output_contract: OutputContract = OutputContract.NUMERIC,
) -> WorkflowRequest:
    config = _config()
    defaults = config.policy_defaults
    return create_workflow_request(
        request_id=request_id,
        messages=(
            ChatMessage(role="system", content="Return an exact numeric result."),
            ChatMessage(role="user", content=expression),
        ),
        output_contract=output_contract,
        policy=RequestPolicyContext(
            data_classification=defaults.data_classification,
            network_allowed=defaults.network_allowed,
            allowed_corpus_ids=defaults.allowed_corpus_ids,
            allowed_tool_ids=defaults.allowed_tool_ids,
            required_quality=defaults.required_quality,
            allow_unknown_output_safety=defaults.allow_unknown_output_safety,
        ),
    )


def _action_and_availability() -> tuple[WorkflowAction, ActionAvailability]:
    registered = build_action_registry(_config()).actions["tool.calculator.v1"]
    availability = ActionAvailability(
        action_id=registered.action.action_id,
        action_fingerprint=registered.action.action_fingerprint,
        status=AvailabilityStatus.AVAILABLE,
    )
    return registered.action, availability


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", "7"),
        ("0.1 + 0.2", "0.3"),
        ("1 / 2", "0.5"),
        ("1 / 3", "1/3"),
        ("-0.125 * 8", "-1"),
        ("2 ** -3", "0.125"),
        ("0x10 + 1_000", "1016"),
    ],
)
def test_evaluates_exact_arithmetic(expression: str, expected: str) -> None:
    result = evaluate_calculator_expression(expression, _config().calculator)

    assert result.status is CalculatorStatus.SUPPORTED
    assert result.response == expected
    assert result.reason_code is None


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("", CalculatorReason.EMPTY_EXPRESSION),
        ("not valid arithmetic", CalculatorReason.INVALID_SYNTAX),
        ("value + 1", CalculatorReason.UNSUPPORTED_SYNTAX),
        ("abs(1)", CalculatorReason.UNSUPPORTED_SYNTAX),
        ("(1).__class__", CalculatorReason.UNSUPPORTED_SYNTAX),
        ("[1, 2]", CalculatorReason.UNSUPPORTED_SYNTAX),
        ("{'value': 1}", CalculatorReason.UNSUPPORTED_SYNTAX),
        ("True + 1", CalculatorReason.UNSUPPORTED_SYNTAX),
        ("1 + 2j", CalculatorReason.UNSUPPORTED_SYNTAX),
        ("1 // 2", CalculatorReason.OPERATOR_NOT_ALLOWED),
        ("1 % 2", CalculatorReason.OPERATOR_NOT_ALLOWED),
        ("1 / 0", CalculatorReason.DIVISION_BY_ZERO),
        ("4 ** 0.5", CalculatorReason.NON_INTEGER_EXPONENT),
        ("2 ** 13", CalculatorReason.EXPONENT_OUT_OF_BOUNDS),
        ("1000000000001", CalculatorReason.LITERAL_OUT_OF_BOUNDS),
        ("1000000000000 + 1", CalculatorReason.RESULT_OUT_OF_BOUNDS),
        ("1e-13", CalculatorReason.LITERAL_EXPONENT_OUT_OF_BOUNDS),
    ],
)
def test_rejects_unsafe_or_out_of_bounds_expressions(
    expression: str,
    reason: CalculatorReason,
) -> None:
    result = evaluate_calculator_expression(expression, _config().calculator)

    assert result.status is CalculatorStatus.UNSUPPORTED
    assert result.response is None
    assert result.reason_code is reason


def test_enforces_length_depth_operation_and_operator_bounds() -> None:
    base = _config().calculator
    too_long = "1" * (base.max_expression_length + 1)
    assert evaluate_calculator_expression(too_long, base).reason_code is (
        CalculatorReason.EXPRESSION_TOO_LONG
    )

    shallow = base.model_copy(update={"max_ast_depth": 2})
    assert evaluate_calculator_expression("1 + (2 + 3)", shallow).reason_code is (
        CalculatorReason.AST_TOO_DEEP
    )

    few_operations = base.model_copy(update={"max_ast_depth": 64, "max_operations": 2})
    assert evaluate_calculator_expression("1 + 2 + 3 + 4", few_operations).reason_code is (
        CalculatorReason.TOO_MANY_OPERATIONS
    )

    no_power = base.model_copy(
        update={
            "operators": tuple(
                operator for operator in base.operators if operator is not CalculatorOperator.POWER
            )
        }
    )
    assert evaluate_calculator_expression("2 ** 3", no_power).reason_code is (
        CalculatorReason.OPERATOR_NOT_ALLOWED
    )


def test_request_requires_numeric_output_contract() -> None:
    result = evaluate_calculator_request(
        _request("1 + 1", output_contract=OutputContract.FREE_TEXT),
        _config().calculator,
    )

    assert result.status is CalculatorStatus.UNSUPPORTED
    assert result.reason_code is CalculatorReason.OUTPUT_CONTRACT_NOT_NUMERIC


def test_suite_metrics_separate_abstention_from_attempted_accuracy() -> None:
    requests = [
        _request("1 + 1", request_id="correct"),
        _request("2 + 2", request_id="wrong"),
        _request("What is three plus three?", request_id="unsupported"),
    ]
    expected = {"correct": "2", "wrong": "5", "unsupported": "6"}

    metrics = evaluate_calculator_suite(requests, expected, _config().calculator)

    assert metrics.total_requests == 3
    assert metrics.supported_requests == 2
    assert metrics.unsupported_requests == 1
    assert metrics.coverage == pytest.approx(2 / 3)
    assert metrics.exact_matches == 1
    assert metrics.supported_accuracy == 0.5
    assert metrics.overall_accuracy == pytest.approx(1 / 3)


def test_natural_language_math_requests_are_explicit_abstentions() -> None:
    requests = [
        _request(
            "Problem:\nA shop has 12 apples and sells 5. How many remain?\n\n"
            "Solve the problem. End with the final numeric answer.",
            request_id="word-problem-1",
        ),
        _request("What is two plus three?", request_id="word-problem-2"),
    ]

    coverage = measure_calculator_coverage(requests, _config().calculator)

    assert coverage.total_requests == 2
    assert coverage.supported_requests == 0
    assert coverage.unsupported_requests == 2
    assert coverage.coverage == 0


def test_parser_eligibility_feeds_policy_availability() -> None:
    config = _config()
    request = _request("What is two plus three?")
    registered = build_action_registry(config).actions["tool.calculator.v1"]
    runtime = ActionRuntimeContext(
        local_model_ready=True,
        verified_adapter_ids=(),
        available_corpus_ids=(),
        replay_action_ids=(),
        calculator_supported=calculator_supported(request, config.calculator),
    )

    availability = evaluate_action_availability(request, registered, runtime)

    assert availability.status is AvailabilityStatus.UNAVAILABLE
    assert availability.reason_code == "calculator_input_unsupported"
    assert availability.rule_ids == (RULE_CALCULATOR_INPUT,)


def test_executor_returns_completed_exact_outcome_with_unknown_compute_cost() -> None:
    config = _config()
    request = _request("0.1 + 0.2")
    action, availability = _action_and_availability()

    outcome = execute_calculator(
        request,
        action,
        availability,
        config.calculator,
    )

    assert outcome.status is ActionOutcomeStatus.COMPLETED
    assert outcome.response == "0.3"
    assert outcome.safety.status is SafetyStatus.SAFE
    assert outcome.telemetry is not None
    assert outcome.telemetry.wall_latency_ms >= 0
    assert outcome.telemetry.provider_fee_usd == 0
    assert outcome.telemetry.compute_cost_usd is None
    assert outcome.telemetry.total_cost_usd is None
    assert outcome.telemetry.energy_provenance is EnergyProvenance.UNAVAILABLE
    assert outcome.metadata["calculator_config_fingerprint"] == (
        calculator_config_fingerprint(config.calculator)
    )


def test_executor_defensively_returns_unsupported_without_a_response() -> None:
    config = _config()
    request = _request("abs(1)")
    action, availability = _action_and_availability()

    outcome = execute_calculator(
        request,
        action,
        availability,
        config.calculator,
    )

    assert outcome.status is ActionOutcomeStatus.UNSUPPORTED
    assert outcome.response is None
    assert outcome.safety.status is SafetyStatus.NOT_ASSESSED
    assert outcome.metadata["reason_code"] == CalculatorReason.UNSUPPORTED_SYNTAX.value
    assert outcome.telemetry is not None


def test_executor_refuses_policy_blocked_or_unavailable_invocation() -> None:
    request = _request("1 + 1")
    action, availability = _action_and_availability()
    unavailable = availability.model_copy(
        update={
            "status": AvailabilityStatus.UNAVAILABLE,
            "reason_code": "calculator_input_unsupported",
            "rule_ids": (RULE_CALCULATOR_INPUT,),
        }
    )

    with pytest.raises(ValueError, match="must not be executed"):
        execute_calculator(
            request,
            action,
            unavailable,
            _config().calculator,
        )
