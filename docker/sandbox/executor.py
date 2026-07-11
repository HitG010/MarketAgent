"""Execute one candidate inside the already-constrained sandbox container."""

from __future__ import annotations

import contextlib
import io
import json
import os
import resource
import traceback
from pathlib import Path
from typing import Any


class BoundedTextBuffer(io.TextIOBase):
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._parts: list[str] = []
        self._size = 0
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        remaining = max(0, self._limit - self._size)
        if remaining:
            kept = value[:remaining]
            self._parts.append(kept)
            self._size += len(kept)
        if len(value) > remaining:
            self.truncated = True
        return len(value)

    def getvalue(self) -> str:
        return "".join(self._parts)


def _set_process_limits(timeout_seconds: float) -> None:
    cpu_seconds = max(1, int(timeout_seconds) + 1)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def _error_text(error: BaseException) -> str:
    return "".join(traceback.format_exception(error, limit=5))


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload["candidate"]
    test_setup = payload["test_setup"]
    tests = payload["tests"]
    timeout_seconds = float(payload["timeout_seconds"])
    output_limit = int(payload["output_limit"])
    if not isinstance(candidate, str) or not isinstance(test_setup, str):
        raise TypeError("candidate and test_setup must be strings")
    if not isinstance(tests, list) or not all(isinstance(test, str) for test in tests):
        raise TypeError("tests must be a list of strings")
    if not tests:
        raise ValueError("at least one test is required")
    if not 0 < output_limit <= 65_536:
        raise ValueError("invalid output limit")

    _set_process_limits(timeout_seconds)
    output = BoundedTextBuffer(output_limit)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    status = "passed"
    error = ""

    try:
        compile(candidate, "<candidate>", "exec")
    except SyntaxError as syntax_error:
        status = "syntax_error"
        error = _error_text(syntax_error)
    else:
        try:
            compile(test_setup, "<test_setup>", "exec")
            for index, test in enumerate(tests):
                compile(test, f"<test_{index}>", "exec")
            compiled_program = compile(
                "\n\n".join([candidate, test_setup, *tests]),
                "<sandbox>",
                "exec",
            )
        except BaseException as test_error:
            status = "infrastructure_error"
            error = _error_text(test_error)
        else:
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(compiled_program, namespace)
            except AssertionError as assertion_error:
                status = "assertion_failure"
                error = _error_text(assertion_error)
            except BaseException as runtime_error:
                status = "runtime_error"
                error = _error_text(runtime_error)

    return {
        "status": status,
        "stdout": output.getvalue(),
        "stderr": error,
        "output_truncated": output.truncated or len(error) > output_limit,
    }


def main() -> int:
    result_path_value = os.environ.get("SMS_RESULT_PATH")
    if not result_path_value:
        return 2
    result_path = Path(result_path_value)
    try:
        payload = json.loads(input())
        result = _execute(payload)
    except BaseException as error:
        result = {
            "status": "infrastructure_error",
            "stdout": "",
            "stderr": _error_text(error),
            "output_truncated": False,
        }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
