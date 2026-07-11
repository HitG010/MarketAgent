"""Host-side Docker boundary for executing untrusted benchmark code."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import threading
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

from pydantic import Field

from small_models_society.schemas import StrictModel

DEFAULT_IMAGE = "small-models-society-sandbox:0.1.0"
DEFAULT_CONTEXT = Path(__file__).resolve().parents[2] / "docker" / "sandbox"


class SandboxStatus(StrEnum):
    PASSED = "passed"
    ASSERTION_FAILURE = "assertion_failure"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class SandboxResult(StrictModel):
    status: SandboxStatus
    duration_ms: float = Field(ge=0)
    stdout: str = ""
    stderr: str = ""
    output_truncated: bool = False

    @property
    def passed(self) -> bool:
        return self.status is SandboxStatus.PASSED


def docker_command(
    docker_executable: str,
    image: str,
    container_name: str,
    memory_mb: int,
    cpus: float,
) -> list[str]:
    """Build the auditable, mount-free Docker command for one candidate."""

    return [
        docker_executable,
        "run",
        "--rm",
        "--interactive",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        f"{memory_mb}m",
        "--memory-swap",
        f"{memory_mb}m",
        "--cpus",
        str(cpus),
        "--user",
        "65532:65532",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=16m",
        "--ulimit",
        "nofile=64:64",
        "--ulimit",
        "nproc=64:64",
        "--ipc",
        "none",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        image,
    ]


def cleanup_command(docker_executable: str, container_name: str) -> list[str]:
    """Build the forced cleanup command for an interrupted container run."""

    return [docker_executable, "rm", "--force", container_name]


class DockerSandbox:
    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        timeout_seconds: float = 2.0,
        output_limit: int = 8_192,
        memory_mb: int = 128,
        cpus: float = 0.5,
        docker_executable: str = "docker",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 256 <= output_limit <= 65_536:
            raise ValueError("output_limit must be between 256 and 65536")
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.output_limit = output_limit
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.docker_executable = docker_executable

    def _infrastructure_error(self, started: float, message: str) -> SandboxResult:
        return SandboxResult(
            status=SandboxStatus.INFRASTRUCTURE_ERROR,
            duration_ms=(time.perf_counter() - started) * 1000,
            stderr=message[: self.output_limit],
            output_truncated=len(message) > self.output_limit,
        )

    @staticmethod
    def _read_bounded(stream: BinaryIO, destination: bytearray, limit: int) -> None:
        while chunk := stream.read(8_192):
            remaining = limit - len(destination)
            if remaining > 0:
                destination.extend(chunk[:remaining])

    def _force_remove_container(self, container_name: str) -> None:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                cleanup_command(self.docker_executable, container_name),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )

    def run(self, candidate: str, test_setup: str, tests: list[str]) -> SandboxResult:
        """Execute candidate code in Docker; this method never invokes Python on the host."""

        started = time.perf_counter()
        if not tests:
            return self._infrastructure_error(started, "at least one test is required")
        container_name = f"sms-sandbox-{uuid.uuid4().hex}"
        command = docker_command(
            self.docker_executable,
            self.image,
            container_name,
            self.memory_mb,
            self.cpus,
        )
        payload = json.dumps(
            {
                "candidate": candidate,
                "test_setup": test_setup,
                "tests": tests,
                "timeout_seconds": self.timeout_seconds,
                "output_limit": self.output_limit,
            },
            separators=(",", ":"),
        ).encode()

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            return self._infrastructure_error(started, f"Docker could not start: {error}")

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        stdout = bytearray()
        stderr = bytearray()
        stdout_thread = threading.Thread(
            target=self._read_bounded,
            args=(process.stdout, stdout, self.output_limit),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_bounded,
            args=(process.stderr, stderr, self.output_limit),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            process.stdin.write(payload)
            process.stdin.close()
            process.wait(timeout=self.timeout_seconds + 5)
        except (BrokenPipeError, OSError) as error:
            self._force_remove_container(container_name)
            with contextlib.suppress(OSError):
                process.kill()
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                process.wait(timeout=5)
            return self._infrastructure_error(started, f"Docker communication failed: {error}")
        except subprocess.TimeoutExpired:
            self._force_remove_container(container_name)
            with contextlib.suppress(OSError):
                process.kill()
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                process.wait(timeout=5)
            return SandboxResult(
                status=SandboxStatus.TIMEOUT,
                duration_ms=(time.perf_counter() - started) * 1000,
                stderr="host timeout terminated the sandbox container",
            )
        except BaseException:
            self._force_remove_container(container_name)
            with contextlib.suppress(OSError):
                process.kill()
            raise
        finally:
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            process.stdout.close()
            process.stderr.close()

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if process.returncode != 0:
            return self._infrastructure_error(
                started,
                f"Docker exited with code {process.returncode}: {stderr_text or stdout_text}",
            )
        try:
            result = SandboxResult.model_validate_json(stdout_text)
        except ValueError as error:
            return self._infrastructure_error(
                started, f"invalid sandbox response: {error}; stderr={stderr_text}"
            )
        return result


def docker_available(docker_executable: str = "docker") -> bool:
    if shutil.which(docker_executable) is None:
        return False
    try:
        result = subprocess.run(
            [docker_executable, "info", "--format", "{{.ServerVersion}}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def sandbox_image_available(image: str = DEFAULT_IMAGE, docker_executable: str = "docker") -> bool:
    if not docker_available(docker_executable):
        return False
    try:
        result = subprocess.run(
            [docker_executable, "image", "inspect", image],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def build_sandbox_image(
    image: str = DEFAULT_IMAGE,
    context: Path = DEFAULT_CONTEXT,
    docker_executable: str = "docker",
) -> None:
    """Build the pinned sandbox image without running candidate code."""

    subprocess.run(
        [docker_executable, "build", "--tag", image, str(context)],
        check=True,
    )
