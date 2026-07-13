from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.inference.adapters import (
    AdapterCatalog,
    AdapterInferenceModules,
    PeftHuggingFaceBackend,
    load_adapter_catalog,
)
from small_models_society.inference.config import load_inference_config
from small_models_society.inference.contracts import ChatMessage, GenerationRequest
from small_models_society.inference.hardware import HardwareReport
from small_models_society.inference.huggingface import InferenceModules
from small_models_society.schemas import Domain
from small_models_society.training.config import TrainingConfig, load_training_config
from small_models_society.training.runner import AdapterRunManifest

ROOT = Path(__file__).parents[1]
INFERENCE_CONFIG = ROOT / "configs" / "inference.yaml"
TRAINING_CONFIG = ROOT / "configs" / "training.yaml"
FIXTURE_BENCHMARK = Path(__file__).parent / "fixtures" / "benchmark.jsonl"


def _training_config(root: Path) -> TrainingConfig:
    config = load_training_config(TRAINING_CONFIG)
    output = config.output.model_copy(update={"adapter_root": str(root)})
    return config.model_copy(update={"output": output})


def _completed_manifest(
    config: TrainingConfig,
    domain: Domain,
    adapter_sha256: str,
) -> AdapterRunManifest:
    values: dict[str, object] = {
        "schema_version": 1,
        "training_config_fingerprint": config.fingerprint(),
        "sft_manifest_sha256": "1" * 64,
        "sft_train_sha256": "2" * 64,
        "sft_validation_sha256": "3" * 64,
        "prompt_catalog_fingerprint": "4" * 64,
        "model_id": config.model.model_id,
        "model_revision": config.model.revision,
        "specialist": domain,
        "selected_device": "cuda",
        "selected_dtype": "bfloat16",
        "python_version": "3.11.9",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "package_versions": {"peft": "0.19.1"},
        "implementation_commit": None,
        "implementation_version": 1,
        "train_source_ids": [f"{domain.value}-train"],
        "validation_source_ids": [f"{domain.value}-validation"],
        "status": "completed",
        "adapter_sha256": adapter_sha256,
        "train_metrics": {"train_loss": 1.0},
        "eval_metrics": {"eval_loss": 0.5},
        "trainable_parameters": 16,
        "total_parameters": 1_016,
        "duration_seconds": 2.5,
        "resumed_from_checkpoint": None,
    }
    fingerprint_values = {
        key: value
        for key, value in values.items()
        if key
        not in {
            "status",
            "adapter_sha256",
            "train_metrics",
            "eval_metrics",
            "trainable_parameters",
            "total_parameters",
            "duration_seconds",
            "resumed_from_checkpoint",
        }
    }
    values["run_fingerprint"] = sha256_bytes(canonical_json(fingerprint_values).encode())
    return AdapterRunManifest.model_validate(values)


def _write_adapter(root: Path, config: TrainingConfig, domain: Domain) -> None:
    path = root / domain.value
    path.mkdir(parents=True)
    weights = f"weights-{domain.value}".encode()
    weights_sha256 = sha256_bytes(weights)
    (path / "adapter_model.safetensors").write_bytes(weights)
    adapter_config = {
        "base_model_name_or_path": config.model.model_id,
        "revision": config.model.revision,
        "r": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": config.lora.bias,
        "task_type": config.lora.task_type,
        "target_modules": [module.value for module in config.lora.target_modules],
    }
    (path / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    manifest = _completed_manifest(config, domain, weights_sha256)
    (path / "manifest.json").write_text(
        canonical_json(manifest.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )


def _catalog(tmp_path: Path) -> tuple[AdapterCatalog, TrainingConfig]:
    training = _training_config(tmp_path)
    for domain in Domain:
        _write_adapter(tmp_path, training, domain)
    return load_adapter_catalog(
        tmp_path,
        training,
        load_inference_config(INFERENCE_CONFIG),
    ), training


def test_loads_four_verified_adapter_specs(tmp_path: Path) -> None:
    catalog, training = _catalog(tmp_path)

    assert set(catalog.adapters) == set(Domain)
    assert catalog.model_id == training.model.model_id
    assert all(len(adapter.sha256) == 64 for adapter in catalog.adapters.values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("weights", "hash does not match"),
        ("revision", "configuration mismatch"),
        ("rank", "configuration mismatch"),
        ("targets", "target modules"),
    ],
)
def test_rejects_corrupt_or_incompatible_adapters(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    training = _training_config(tmp_path)
    for domain in Domain:
        _write_adapter(tmp_path, training, domain)
    math_path = tmp_path / "math"
    if mutation == "weights":
        (math_path / "adapter_model.safetensors").write_bytes(b"tampered")
    else:
        config = json.loads((math_path / "adapter_config.json").read_text(encoding="utf-8"))
        if mutation == "revision":
            config["revision"] = "a" * 40
        elif mutation == "rank":
            config["r"] = 16
        else:
            config["target_modules"] = ["q_proj"]
        (math_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_adapter_catalog(
            tmp_path,
            training,
            load_inference_config(INFERENCE_CONFIG),
        )


class FakeVector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int]:
        return (len(self.values),)

    def __getitem__(self, item: slice) -> FakeVector:
        return FakeVector(self.values[item])

    def tolist(self) -> list[int]:
        return self.values


class FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.values))

    def __getitem__(self, item: object) -> FakeVector:
        if isinstance(item, int):
            return FakeVector(self.values)
        if isinstance(item, tuple) and isinstance(item[-1], slice):
            return FakeVector(self.values[item[-1]])
        raise TypeError(item)

    def to(self, _device: str) -> FakeTensor:
        return self


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, _messages: object, **_kwargs: object) -> str:
        return "prompt"

    def __call__(self, text: str, **_kwargs: object) -> dict[str, FakeTensor]:
        return {"input_ids": FakeTensor(list(range(len(text))))}

    def decode(self, _ids: object, **_kwargs: object) -> str:
        return "answer"


class FakeBaseModel:
    def to(self, _device: str) -> FakeBaseModel:
        return self

    def eval(self) -> None:
        pass


class FakePeftModel(FakeBaseModel):
    def __init__(self) -> None:
        self.loaded: list[tuple[Path, str]] = []
        self.active: list[str | None] = []
        self.eval_calls = 0

    def load_adapter(self, path: Path, *, adapter_name: str, **kwargs: object) -> None:
        assert kwargs == {"is_trainable": False, "low_cpu_mem_usage": True}
        self.loaded.append((path, adapter_name))

    def set_adapter(self, name: str) -> None:
        self.active.append(name)

    @contextlib.contextmanager
    def disable_adapter(self) -> Any:
        self.active.append(None)
        yield

    def generate(self, **kwargs: Any) -> FakeTensor:
        prompt = kwargs["input_ids"].values
        return FakeTensor([*prompt, 7, 99])

    def eval(self) -> None:
        self.eval_calls += 1


class FakePeftFactory:
    def __init__(self, model: FakePeftModel) -> None:
        self.model = model
        self.calls: list[tuple[object, Path, dict[str, object]]] = []

    def from_pretrained(
        self,
        base: object,
        path: Path,
        **kwargs: object,
    ) -> FakePeftModel:
        self.calls.append((base, path, kwargs))
        return self.model


class FakeFactory:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def from_pretrained(self, _model_id: str, **_kwargs: object) -> object:
        self.calls += 1
        return self.value


class FakeTorch:
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    __version__ = "test"
    cuda = SimpleNamespace(manual_seed_all=lambda _seed: None, synchronize=lambda: None)

    def manual_seed(self, _seed: int) -> None:
        pass

    def inference_mode(self) -> Any:
        return contextlib.nullcontext()


def _hardware() -> HardwareReport:
    return HardwareReport(
        ready=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        python_version="3.11.9",
        package_versions={"torch": "test"},
        selected_device="cpu",
        selected_dtype="float32",
        cuda_available=False,
        system_ram_gb=16,
        model_cache_path="C:/cache/model",
        model_cached=True,
    )


def _request(adapter: str | None) -> GenerationRequest:
    from small_models_society.data.prepare import load_benchmark
    from small_models_society.inference.contracts import to_inference_example

    example = to_inference_example(load_benchmark(FIXTURE_BENCHMARK)[0])
    return GenerationRequest(
        request_id=example.id,
        profile="general",
        adapter=adapter,
        messages=[
            ChatMessage(role="system", content="General"),
            ChatMessage(role="user", content="Question"),
        ],
        max_new_tokens=2,
    )


def test_loads_base_once_switches_adapters_and_disables_for_base(tmp_path: Path) -> None:
    catalog, _training = _catalog(tmp_path)
    tokenizer_factory = FakeFactory(FakeTokenizer())
    base_factory = FakeFactory(FakeBaseModel())
    peft_model = FakePeftModel()
    peft_factory = FakePeftFactory(peft_model)
    modules = AdapterInferenceModules(
        inference=InferenceModules(
            torch=FakeTorch(),
            auto_tokenizer=tokenizer_factory,
            auto_model=base_factory,
            torch_version="test",
            transformers_version="test",
        ),
        peft_model=peft_factory,
        peft_version="test-peft",
    )
    times = iter([1.0, 1.1, 2.0, 2.1])
    backend = PeftHuggingFaceBackend(
        load_inference_config(INFERENCE_CONFIG),
        _hardware(),
        catalog,
        modules,
        clock=lambda: next(times),
    )

    adapted = backend.generate(_request("math"))
    base = backend.generate(_request(None))

    assert tokenizer_factory.calls == 1
    assert base_factory.calls == 1
    assert peft_factory.calls[0][2] == {
        "adapter_name": "math",
        "is_trainable": False,
        "low_cpu_mem_usage": True,
    }
    assert [name for _path, name in peft_model.loaded] == ["code", "logic", "knowledge"]
    assert peft_model.active == ["math", None]
    assert adapted.metadata["adapter"] == "math"
    assert adapted.metadata["adapter_sha256"] == catalog.adapters[Domain.MATH].sha256
    assert base.metadata["adapter"] is None
    assert base.metadata["adapter_sha256"] is None
    assert adapted.metadata["peft_version"] == "test-peft"


def test_base_backend_rejects_adapter_request() -> None:
    from small_models_society.inference.huggingface import HuggingFaceBackend

    backend = object.__new__(HuggingFaceBackend)

    with pytest.raises(ValueError, match="does not support adapters"):
        backend._activate_request(_request("math"))
