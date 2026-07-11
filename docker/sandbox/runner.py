"""Trusted PID 1 wrapper that enforces timeout and emits one bounded result."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

MAX_PAYLOAD_BYTES = 1_048_576


def _bounded(value: bytes, limit: int) -> tuple[str, bool]:
    truncated = len(value) > limit
    return value[:limit].decode("utf-8", errors="replace"), truncated


def _run(payload: dict[str, Any], raw_payload: bytes) -> dict[str, Any]:
    timeout_seconds = float(payload["timeout_seconds"])
    output_limit = int(payload["output_limit"])
    if not 0.05 <= timeout_seconds <= 30:
        raise ValueError("timeout_seconds must be between 0.05 and 30")
    if not 256 <= output_limit <= 65_536:
        raise ValueError("output_limit must be between 256 and 65536")

    with tempfile.TemporaryDirectory(prefix="sms-") as temporary_directory:
        result_path = Path(temporary_directory) / "result.json"
        environment = {**os.environ, "SMS_RESULT_PATH": str(result_path)}
        process = subprocess.Popen(
            [sys.executable, "-I", "/sandbox/executor.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=environment,
        )
        try:
            child_stdout, child_stderr = process.communicate(
                input=raw_payload + b"\n", timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            child_stdout, child_stderr = process.communicate()
            stdout, stdout_truncated = _bounded(child_stdout, output_limit)
            stderr, stderr_truncated = _bounded(child_stderr, output_limit)
            return {
                "status": "timeout",
                "stdout": stdout,
                "stderr": stderr,
                "output_truncated": stdout_truncated or stderr_truncated,
            }

        if process.returncode != 0 or not result_path.exists():
            stdout, stdout_truncated = _bounded(child_stdout, output_limit)
            stderr, stderr_truncated = _bounded(child_stderr, output_limit)
            return {
                "status": "runtime_error",
                "stdout": stdout,
                "stderr": stderr or f"executor exited with code {process.returncode}",
                "output_truncated": stdout_truncated or stderr_truncated,
            }

        result = json.loads(result_path.read_text(encoding="utf-8"))
        bypass_output, bypass_truncated = _bounded(child_stdout + child_stderr, output_limit)
        if bypass_output:
            result["stdout"] = (result.get("stdout", "") + bypass_output)[:output_limit]
        result["output_truncated"] = bool(result.get("output_truncated", False) or bypass_truncated)
        result["stderr"] = str(result.get("stderr", ""))[:output_limit]
        return result


def main() -> int:
    started = time.perf_counter()
    raw_payload = sys.stdin.buffer.read(MAX_PAYLOAD_BYTES + 1)
    try:
        if len(raw_payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("payload exceeds one MiB")
        payload = json.loads(raw_payload)
        if not isinstance(payload, dict):
            raise TypeError("payload must be an object")
        result = _run(payload, raw_payload)
    except BaseException as error:
        result = {
            "status": "infrastructure_error",
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "output_truncated": False,
        }
    result["duration_ms"] = (time.perf_counter() - started) * 1000
    sys.stdout.write(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
