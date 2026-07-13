"""Bounded exact-arithmetic calculator without dynamic code execution."""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from fractions import Fraction
from time import perf_counter
from typing import NoReturn, Self

from pydantic import Field, model_validator

from small_models_society.data.prepare import canonical_json
from small_models_society.routing.config import (
    ActionKind,
    CalculatorConfig,
    CalculatorOperator,
)
from small_models_society.routing.contracts import (
    ActionAvailability,
    ActionOutcome,
    ActionOutcomeStatus,
    ActionTelemetry,
    AvailabilityStatus,
    EnergyProvenance,
    OutputContract,
    SafetyAssessment,
    SafetyStatus,
    WorkflowAction,
    WorkflowRequest,
)
from small_models_society.schemas import StrictModel

CALCULATOR_EXECUTOR_ID = "calculator.ast.v1"


def calculator_config_fingerprint(config: CalculatorConfig) -> str:
    return hashlib.sha256(
        canonical_json(config.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


class CalculatorStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class CalculatorReason(StrEnum):
    EMPTY_EXPRESSION = "empty_expression"
    EXPRESSION_TOO_LONG = "expression_too_long"
    INVALID_SYNTAX = "invalid_syntax"
    UNSUPPORTED_SYNTAX = "unsupported_syntax"
    OUTPUT_CONTRACT_NOT_NUMERIC = "output_contract_not_numeric"
    OPERATOR_NOT_ALLOWED = "operator_not_allowed"
    AST_TOO_DEEP = "ast_too_deep"
    TOO_MANY_OPERATIONS = "too_many_operations"
    LITERAL_EXPONENT_OUT_OF_BOUNDS = "literal_exponent_out_of_bounds"
    LITERAL_OUT_OF_BOUNDS = "literal_out_of_bounds"
    NON_INTEGER_EXPONENT = "non_integer_exponent"
    EXPONENT_OUT_OF_BOUNDS = "exponent_out_of_bounds"
    DIVISION_BY_ZERO = "division_by_zero"
    RESULT_OUT_OF_BOUNDS = "result_out_of_bounds"


class CalculatorEvaluation(StrictModel):
    status: CalculatorStatus
    response: str | None = None
    reason_code: CalculatorReason | None = None

    @model_validator(mode="after")
    def require_status_details(self) -> Self:
        if self.status is CalculatorStatus.SUPPORTED:
            if self.response is None or self.reason_code is not None:
                raise ValueError("supported calculator result requires only a response")
        elif self.response is not None or self.reason_code is None:
            raise ValueError("unsupported calculator result requires only a reason")
        return self


class CalculatorCoverage(StrictModel):
    total_requests: int = Field(gt=0)
    supported_requests: int = Field(ge=0)
    unsupported_requests: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.supported_requests + self.unsupported_requests != self.total_requests:
            raise ValueError("calculator coverage counts must sum to total_requests")
        expected = self.supported_requests / self.total_requests
        if not math.isclose(self.coverage, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("calculator coverage does not match counts")
        return self


class CalculatorSuiteMetrics(CalculatorCoverage):
    exact_matches: int = Field(ge=0)
    supported_accuracy: float | None = Field(default=None, ge=0, le=1)
    overall_accuracy: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_accuracy(self) -> Self:
        if self.exact_matches > self.supported_requests:
            raise ValueError("exact calculator matches cannot exceed supported requests")
        expected_supported = (
            self.exact_matches / self.supported_requests if self.supported_requests else None
        )
        if expected_supported is None:
            if self.supported_accuracy is not None:
                raise ValueError("supported accuracy must be null with no supported requests")
        elif self.supported_accuracy is None or not math.isclose(
            self.supported_accuracy,
            expected_supported,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("supported calculator accuracy does not match counts")
        expected_overall = self.exact_matches / self.total_requests
        if not math.isclose(
            self.overall_accuracy,
            expected_overall,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("overall calculator accuracy does not match counts")
        return self


class _UnsupportedExpression(Exception):
    def __init__(self, reason: CalculatorReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _unsupported(reason: CalculatorReason) -> NoReturn:
    raise _UnsupportedExpression(reason)


def _fraction_from_decimal_literal(
    expression: str,
    node: ast.Constant,
    config: CalculatorConfig,
) -> Fraction:
    segment = ast.get_source_segment(expression, node)
    if segment is None:
        _unsupported(CalculatorReason.UNSUPPORTED_SYNTAX)
    try:
        decimal_value = Decimal(segment.replace("_", ""))
    except InvalidOperation:
        _unsupported(CalculatorReason.UNSUPPORTED_SYNTAX)
    if not decimal_value.is_finite():
        _unsupported(CalculatorReason.LITERAL_OUT_OF_BOUNDS)
    decimal_exponent = decimal_value.as_tuple().exponent
    if not isinstance(decimal_exponent, int) or abs(decimal_exponent) > config.max_exponent:
        _unsupported(CalculatorReason.LITERAL_EXPONENT_OUT_OF_BOUNDS)
    return Fraction(decimal_value)


def _calculator_operator(node: ast.operator) -> CalculatorOperator | None:
    if isinstance(node, ast.Add):
        return CalculatorOperator.ADD
    if isinstance(node, ast.Sub):
        return CalculatorOperator.SUBTRACT
    if isinstance(node, ast.Mult):
        return CalculatorOperator.MULTIPLY
    if isinstance(node, ast.Div):
        return CalculatorOperator.DIVIDE
    if isinstance(node, ast.Pow):
        return CalculatorOperator.POWER
    return None


class _FractionEvaluator:
    def __init__(self, expression: str, config: CalculatorConfig) -> None:
        self.expression = expression
        self.config = config
        self.operation_count = 0
        self.max_abs_value = Fraction(Decimal(str(config.max_abs_value)))

    def _check_depth(self, depth: int) -> None:
        if depth > self.config.max_ast_depth:
            _unsupported(CalculatorReason.AST_TOO_DEEP)

    def _count_operation(self) -> None:
        self.operation_count += 1
        if self.operation_count > self.config.max_operations:
            _unsupported(CalculatorReason.TOO_MANY_OPERATIONS)

    def _check_value(
        self,
        value: Fraction,
        reason: CalculatorReason = CalculatorReason.RESULT_OUT_OF_BOUNDS,
    ) -> Fraction:
        if abs(value) > self.max_abs_value:
            _unsupported(reason)
        return value

    def _constant(self, node: ast.Constant) -> Fraction:
        if isinstance(node.value, bool):
            _unsupported(CalculatorReason.UNSUPPORTED_SYNTAX)
        if isinstance(node.value, int):
            return self._check_value(
                Fraction(node.value),
                CalculatorReason.LITERAL_OUT_OF_BOUNDS,
            )
        if isinstance(node.value, float):
            return self._check_value(
                _fraction_from_decimal_literal(self.expression, node, self.config),
                CalculatorReason.LITERAL_OUT_OF_BOUNDS,
            )
        _unsupported(CalculatorReason.UNSUPPORTED_SYNTAX)

    def evaluate(self, node: ast.AST, depth: int = 1) -> Fraction:
        self._check_depth(depth)
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.UnaryOp):
            self._count_operation()
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                _unsupported(CalculatorReason.OPERATOR_NOT_ALLOWED)
            operand = self.evaluate(node.operand, depth + 1)
            result = operand if isinstance(node.op, ast.UAdd) else -operand
            return self._check_value(result)
        if isinstance(node, ast.BinOp):
            self._count_operation()
            operator = _calculator_operator(node.op)
            if operator is None or operator not in set(self.config.operators):
                _unsupported(CalculatorReason.OPERATOR_NOT_ALLOWED)
            left = self.evaluate(node.left, depth + 1)
            right = self.evaluate(node.right, depth + 1)
            if operator is CalculatorOperator.ADD:
                result = left + right
            elif operator is CalculatorOperator.SUBTRACT:
                result = left - right
            elif operator is CalculatorOperator.MULTIPLY:
                result = left * right
            elif operator is CalculatorOperator.DIVIDE:
                if right == 0:
                    _unsupported(CalculatorReason.DIVISION_BY_ZERO)
                result = left / right
            else:
                if right.denominator != 1:
                    _unsupported(CalculatorReason.NON_INTEGER_EXPONENT)
                exponent = right.numerator
                if abs(exponent) > self.config.max_exponent:
                    _unsupported(CalculatorReason.EXPONENT_OUT_OF_BOUNDS)
                if left == 0 and exponent < 0:
                    _unsupported(CalculatorReason.DIVISION_BY_ZERO)
                result = left**exponent
            return self._check_value(result)
        _unsupported(CalculatorReason.UNSUPPORTED_SYNTAX)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)

    denominator = value.denominator
    twos = 0
    fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"

    scale = max(twos, fives)
    scaled_numerator = value.numerator * (10**scale // value.denominator)
    sign = "-" if scaled_numerator < 0 else ""
    digits = str(abs(scaled_numerator)).rjust(scale + 1, "0")
    whole = digits[:-scale]
    fractional = digits[-scale:].rstrip("0")
    return f"{sign}{whole}.{fractional}" if fractional else f"{sign}{whole}"


def evaluate_calculator_expression(
    expression: str,
    config: CalculatorConfig,
) -> CalculatorEvaluation:
    stripped = expression.strip()
    if not stripped:
        return CalculatorEvaluation(
            status=CalculatorStatus.UNSUPPORTED,
            reason_code=CalculatorReason.EMPTY_EXPRESSION,
        )
    if len(stripped) > config.max_expression_length:
        return CalculatorEvaluation(
            status=CalculatorStatus.UNSUPPORTED,
            reason_code=CalculatorReason.EXPRESSION_TOO_LONG,
        )
    try:
        parsed = ast.parse(stripped, mode="eval")
        value = _FractionEvaluator(stripped, config).evaluate(parsed.body)
    except (SyntaxError, ValueError, RecursionError):
        return CalculatorEvaluation(
            status=CalculatorStatus.UNSUPPORTED,
            reason_code=CalculatorReason.INVALID_SYNTAX,
        )
    except _UnsupportedExpression as error:
        return CalculatorEvaluation(
            status=CalculatorStatus.UNSUPPORTED,
            reason_code=error.reason,
        )
    return CalculatorEvaluation(
        status=CalculatorStatus.SUPPORTED,
        response=_format_fraction(value),
    )


def evaluate_calculator_request(
    request: WorkflowRequest,
    config: CalculatorConfig,
) -> CalculatorEvaluation:
    if request.output_contract is not OutputContract.NUMERIC:
        return CalculatorEvaluation(
            status=CalculatorStatus.UNSUPPORTED,
            reason_code=CalculatorReason.OUTPUT_CONTRACT_NOT_NUMERIC,
        )
    return evaluate_calculator_expression(request.messages[-1].content, config)


def calculator_supported(request: WorkflowRequest, config: CalculatorConfig) -> bool:
    return evaluate_calculator_request(request, config).status is CalculatorStatus.SUPPORTED


def measure_calculator_coverage(
    requests: Sequence[WorkflowRequest],
    config: CalculatorConfig,
) -> CalculatorCoverage:
    if not requests:
        raise ValueError("calculator coverage requires at least one request")
    supported = sum(calculator_supported(request, config) for request in requests)
    return CalculatorCoverage(
        total_requests=len(requests),
        supported_requests=supported,
        unsupported_requests=len(requests) - supported,
        coverage=supported / len(requests),
    )


def evaluate_calculator_suite(
    requests: Sequence[WorkflowRequest],
    expected_outputs: Mapping[str, str],
    config: CalculatorConfig,
) -> CalculatorSuiteMetrics:
    request_ids = [request.request_id for request in requests]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("calculator suite contains duplicate request IDs")
    if set(request_ids) != set(expected_outputs):
        raise ValueError("calculator requests and expected outputs must have identical IDs")

    evaluations = [evaluate_calculator_request(request, config) for request in requests]
    supported = sum(result.status is CalculatorStatus.SUPPORTED for result in evaluations)
    exact_matches = sum(
        result.status is CalculatorStatus.SUPPORTED
        and result.response == expected_outputs[request.request_id]
        for request, result in zip(requests, evaluations, strict=True)
    )
    return CalculatorSuiteMetrics(
        total_requests=len(requests),
        supported_requests=supported,
        unsupported_requests=len(requests) - supported,
        coverage=supported / len(requests),
        exact_matches=exact_matches,
        supported_accuracy=(exact_matches / supported if supported else None),
        overall_accuracy=exact_matches / len(requests),
    )


def execute_calculator(
    request: WorkflowRequest,
    action: WorkflowAction,
    availability: ActionAvailability,
    config: CalculatorConfig,
) -> ActionOutcome:
    if (
        action.kind is not ActionKind.TOOL
        or action.executor_id != CALCULATOR_EXECUTOR_ID
        or action.tool_id != config.tool_id
    ):
        raise ValueError("calculator executor received an incompatible workflow action")
    if availability.status is not AvailabilityStatus.AVAILABLE:
        raise ValueError("blocked or unavailable calculator actions must not be executed")

    started = perf_counter()
    evaluation = evaluate_calculator_request(request, config)
    telemetry = ActionTelemetry(
        wall_latency_ms=(perf_counter() - started) * 1000,
        provider_fee_usd=0,
        compute_cost_usd=None,
        total_cost_usd=None,
        energy_provenance=EnergyProvenance.UNAVAILABLE,
        device="cpu",
    )
    if evaluation.status is CalculatorStatus.SUPPORTED:
        return ActionOutcome(
            request_id=request.request_id,
            request_fingerprint=request.request_fingerprint,
            action_id=action.action_id,
            action_fingerprint=action.action_fingerprint,
            status=ActionOutcomeStatus.COMPLETED,
            availability=availability,
            response=evaluation.response,
            safety=SafetyAssessment(
                status=SafetyStatus.SAFE,
                source=CALCULATOR_EXECUTOR_ID,
            ),
            telemetry=telemetry,
            metadata={
                "executor_id": CALCULATOR_EXECUTOR_ID,
                "calculator_config_fingerprint": calculator_config_fingerprint(config),
            },
        )
    assert evaluation.reason_code is not None
    return ActionOutcome(
        request_id=request.request_id,
        request_fingerprint=request.request_fingerprint,
        action_id=action.action_id,
        action_fingerprint=action.action_fingerprint,
        status=ActionOutcomeStatus.UNSUPPORTED,
        availability=availability,
        safety=SafetyAssessment(
            status=SafetyStatus.NOT_ASSESSED,
            source=CALCULATOR_EXECUTOR_ID,
        ),
        telemetry=telemetry,
        metadata={
            "executor_id": CALCULATOR_EXECUTOR_ID,
            "calculator_config_fingerprint": calculator_config_fingerprint(config),
            "reason_code": evaluation.reason_code.value,
        },
    )
