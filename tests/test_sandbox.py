from __future__ import annotations

from small_models_society.sandbox import SandboxStatus, cleanup_command, docker_command


def test_docker_command_enforces_security_boundaries() -> None:
    command = docker_command("docker", "sandbox:test", "test-container", 128, 0.5)
    rendered = " ".join(command)

    assert "--interactive" in command
    assert "--network none" in rendered
    assert "--read-only" in command
    assert "--cap-drop ALL" in rendered
    assert "--security-opt no-new-privileges" in rendered
    assert "--pids-limit 64" in rendered
    assert "--memory 128m" in rendered
    assert "--memory-swap 128m" in rendered
    assert "--user 65532:65532" in rendered
    assert "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=16m" in rendered
    assert "--ipc none" in rendered
    assert "--volume" not in command
    assert "-v" not in command


def test_sandbox_status_is_stable_string_enum() -> None:
    assert SandboxStatus.PASSED.value == "passed"
    assert SandboxStatus.TIMEOUT.value == "timeout"


def test_interrupted_container_is_force_removed() -> None:
    assert cleanup_command("docker", "test-container") == [
        "docker",
        "rm",
        "--force",
        "test-container",
    ]
