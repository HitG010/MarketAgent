from __future__ import annotations

import pytest

from small_models_society.sandbox import (
    DockerSandbox,
    SandboxStatus,
    build_sandbox_image,
    docker_available,
)

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def sandbox() -> DockerSandbox:
    if not docker_available():
        pytest.skip("Docker daemon is unavailable")
    build_sandbox_image()
    return DockerSandbox(timeout_seconds=2.0)


@pytest.mark.parametrize(
    ("candidate", "tests", "expected_status"),
    [
        ("def add(a, b):\n    return a + b", ["assert add(2, 3) == 5"], SandboxStatus.PASSED),
        (
            "def add(a, b):\n    return a - b",
            ["assert add(2, 3) == 5"],
            SandboxStatus.ASSERTION_FAILURE,
        ),
        ("def broken(:\n    pass", ["assert True"], SandboxStatus.SYNTAX_ERROR),
        ("raise RuntimeError('boom')", ["assert True"], SandboxStatus.RUNTIME_ERROR),
        ("while True:\n    pass", ["assert True"], SandboxStatus.TIMEOUT),
    ],
)
def test_candidate_outcomes(
    sandbox: DockerSandbox,
    candidate: str,
    tests: list[str],
    expected_status: SandboxStatus,
) -> None:
    result = sandbox.run(candidate, "", tests)

    assert result.status is expected_status


def test_network_access_is_blocked(sandbox: DockerSandbox) -> None:
    result = sandbox.run(
        "import socket\ndef network_is_blocked():\n"
        "    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "    try:\n"
        "        connection.connect(('1.1.1.1', 53))\n"
        "    except OSError:\n"
        "        return True\n"
        "    finally:\n"
        "        connection.close()\n"
        "    return False",
        "",
        ["assert network_is_blocked()"],
    )

    assert result.status is SandboxStatus.PASSED


def test_candidate_cannot_bypass_assertions_by_replacing_exec(
    sandbox: DockerSandbox,
) -> None:
    result = sandbox.run(
        "import builtins\nbuiltins.exec = lambda *args, **kwargs: None\n"
        "def add(a, b):\n    return 0",
        "",
        ["assert add(2, 3) == 5"],
    )

    assert result.status is SandboxStatus.ASSERTION_FAILURE
