from __future__ import annotations

import contextlib
import importlib
from pathlib import Path
from typing import Any

import pytest

from small_models_society.data.prepare import load_benchmark
from small_models_society.inference.config import load_inference_config
from small_models_society.inference.contracts import (
    ChatMessage,
    GenerationRequest,
    to_inference_example,
)
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.huggingface import (
    HuggingFaceBackend,
    InferenceDependencyError,
    InferenceModules,
    InferenceOutOfMemoryError,
    load_inference_modules,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "inference.yaml"
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


class FakeVector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int]:
        return (len(self.values),)

    def __getitem__(self, item: slice) -> FakeVector:
        return FakeVector(self.values[item])

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.device: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.values))

    def __getitem__(self, item: object) -> FakeTensor | FakeVector:
        if isinstance(item, int):
            return FakeVector(self.values)
        if isinstance(item, tuple):
            final_item = item[-1]
            if isinstance(final_item, slice):
                return FakeTensor(self.values[final_item])
        raise TypeError(f"unsupported fake tensor index: {item!r}")

    def to(self, device: str) -> FakeTensor:
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.tokenize_calls: list[dict[str, Any]] = []
        self.decoded: list[int] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.chat_calls.append({"messages": messages, **kwargs})
        system = messages[0]["content"]
        user = messages[-1]["content"]
        return f"{system}|{user}|<A>"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, FakeTensor]:
        self.tokenize_calls.append({"text": text, **kwargs})
        values = [ord(character) for character in text]
        return {
            "input_ids": FakeTensor(values),
            "attention_mask": FakeTensor([1] * len(values)),
        }

    def decode(self, token_ids: FakeVector, **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": True}
        self.decoded = token_ids.tolist()
        return "generated response"


class FakeFactory:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
        self.calls.append((model_id, kwargs))
        return self.value


class FakeModel:
    def __init__(self, generated: list[int] | None = None) -> None:
        self.generated = generated or [7, 99]
        self.device: str | None = None
        self.eval_called = False
        self.generate_calls: list[dict[str, Any]] = []
        self.error: RuntimeError | None = None

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: Any) -> FakeTensor:
        self.generate_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        prompt = kwargs["input_ids"].values
        return FakeTensor([*prompt, *self.generated])


class FakeCuda:
    def __init__(self) -> None:
        self.seed: int | None = None
        self.synchronize_calls = 0

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class FakeTorch:
    __version__ = "test-torch"
    float32 = "float32-object"
    float16 = "float16-object"
    bfloat16 = "bfloat16-object"

    def __init__(self) -> None:
        self.seed: int | None = None
        self.cuda = FakeCuda()

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def inference_mode(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


def _hardware(device: str = "cpu", dtype: str = "float32") -> HardwareReport:
    return HardwareReport(
        ready=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        python_version="3.11.9",
        package_versions={"torch": "test-torch", "transformers": "test-transformers"},
        selected_device=device,  # type: ignore[arg-type]
        selected_dtype=dtype,  # type: ignore[arg-type]
        cuda_available=device == "cuda",
        model_cache_path="C:/cache/model",
        model_cached=True,
    )


def _request(max_new_tokens: int = 2) -> GenerationRequest:
    example = to_inference_example(load_benchmark(FIXTURE_BENCHMARK)[0])
    return GenerationRequest(
        request_id=example.id,
        profile="general",
        messages=[
            ChatMessage(role="system", content="Be careful."),
            ChatMessage(role="user", content="Solve this."),
        ],
        max_new_tokens=max_new_tokens,
    )


def _backend(
    generated: list[int] | None = None,
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[HuggingFaceBackend, FakeTokenizer, FakeModel, FakeTorch, FakeFactory, FakeFactory]:
    config = load_inference_config(CONFIG_PATH)
    generation = config.generation.model_copy(update={"max_input_tokens": 24})
    config = config.model_copy(update={"generation": generation})
    tokenizer = FakeTokenizer()
    model = FakeModel(generated)
    torch_module = FakeTorch()
    tokenizer_factory = FakeFactory(tokenizer)
    model_factory = FakeFactory(model)
    modules = InferenceModules(
        torch=torch_module,
        auto_tokenizer=tokenizer_factory,
        auto_model=model_factory,
        torch_version="test-torch",
        transformers_version="test-transformers",
    )
    times = iter([10.0, 10.125])
    backend = HuggingFaceBackend(
        config,
        _hardware(device, dtype),
        modules,
        clock=lambda: next(times),
    )
    return backend, tokenizer, model, torch_module, tokenizer_factory, model_factory


def test_loads_pinned_model_once_with_safe_options() -> None:
    backend, _tokenizer, model, torch_module, tokenizer_factory, model_factory = _backend()

    expected_common = {
        "revision": backend.config.model.revision,
        "trust_remote_code": False,
        "local_files_only": False,
    }
    assert tokenizer_factory.calls == [(backend.config.model.model_id, expected_common)]
    assert model_factory.calls == [
        (
            backend.config.model.model_id,
            {**expected_common, "use_safetensors": True, "dtype": "float32-object"},
        )
    ]
    assert model.device == "cpu"
    assert model.eval_called is True
    assert torch_module.seed == 42


def test_generation_uses_chat_template_greedy_decoding_and_right_truncation() -> None:
    backend, tokenizer, model, _torch, _tokenizer_factory, _model_factory = _backend()

    output = backend.generate(_request())

    assert tokenizer.chat_calls[0]["tokenize"] is False
    assert tokenizer.chat_calls[0]["add_generation_prompt"] is True
    assert all(
        call["return_tensors"] == "pt" and call["add_special_tokens"] is False
        for call in tokenizer.tokenize_calls
    )
    generation_call = model.generate_calls[0]
    assert len(generation_call["input_ids"].values) <= 24
    assert generation_call["input_ids"].values[-3:] == [ord("<"), ord("A"), ord(">")]
    assert generation_call["input_ids"].device == "cpu"
    assert generation_call["do_sample"] is False
    assert generation_call["max_new_tokens"] == 2
    assert generation_call["eos_token_id"] == 99
    assert generation_call["pad_token_id"] == 0
    assert output.text == "generated response"
    assert output.prompt_tokens == 24
    assert output.completion_tokens == 2
    assert output.latency_ms == pytest.approx(125.0)
    assert output.metadata["input_truncated"] is True
    assert output.metadata["original_prompt_tokens"] == len("Be careful.|Solve this.|<A>")
    assert output.metadata["stop_reason"] == "eos"
    assert tokenizer.decoded == [7, 99]


def test_reports_max_token_stop_reason() -> None:
    backend, _tokenizer, _model, _torch, _tokenizer_factory, _model_factory = _backend(
        generated=[7, 8]
    )

    output = backend.generate(_request(max_new_tokens=2))

    assert output.metadata["stop_reason"] == "max_tokens"


def test_cuda_backend_synchronizes_and_seeds_device() -> None:
    backend, _tokenizer, _model, torch_module, _tokenizer_factory, _model_factory = _backend(
        device="cuda", dtype="bfloat16"
    )

    backend.generate(_request())

    assert torch_module.cuda.seed == 42
    assert torch_module.cuda.synchronize_calls == 2


def test_out_of_memory_has_actionable_error() -> None:
    backend, _tokenizer, model, _torch, _tokenizer_factory, _model_factory = _backend()
    model.error = RuntimeError("CUDA out of memory")

    with pytest.raises(InferenceOutOfMemoryError, match="sms inference doctor"):
        backend.generate(_request())


def test_lazy_dependency_error_does_not_require_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)

    with pytest.raises(InferenceDependencyError, match=r"requirements-inference\.lock"):
        load_inference_modules()
