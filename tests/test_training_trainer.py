from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from small_models_society.inference.contracts import ChatMessage
from small_models_society.schemas import Domain
from small_models_society.training.config import TrainingConfig, load_training_config
from small_models_society.training.contracts import SFTTrainingRecord, TrainingSplit
from small_models_society.training.hardware import TrainingHardwareReport
from small_models_society.training.trainer import (
    LoraTrainerBackend,
    TrainingDependencyError,
    TrainingModules,
    TrainingOutOfMemoryError,
    load_training_modules,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "training.yaml"


class FakeParameter:
    def __init__(self, count: int, requires_grad: bool) -> None:
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.count


class FakeTokenizer:
    pad_token_id: int | None = None
    pad_token: str | None = None
    eos_token = "<eos>"


class FakeBaseModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=True)
        self.device: str | None = None
        self.checkpoint_kwargs: dict[str, object] | None = None

    def gradient_checkpointing_enable(self, **kwargs: object) -> None:
        self.checkpoint_kwargs = kwargs

    def to(self, device: str) -> FakeBaseModel:
        self.device = device
        return self


class FakePeftModel:
    def __init__(self, *, save_full_weights: bool = False) -> None:
        self.parameters_value = [
            FakeParameter(1_000, False),
            FakeParameter(16, True),
        ]
        self.save_full_weights = save_full_weights

    def parameters(self) -> list[FakeParameter]:
        return self.parameters_value

    def save_pretrained(self, path: str, **kwargs: object) -> None:
        assert kwargs == {"safe_serialization": True}
        destination = Path(path)
        destination.mkdir(parents=True)
        (destination / "adapter_model.safetensors").write_bytes(b"adapter")
        (destination / "adapter_config.json").write_text("{}", encoding="utf-8")
        if self.save_full_weights:
            (destination / "model.safetensors").write_bytes(b"base")


class FakeFactory:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, model_id: str, **kwargs: object) -> object:
        self.calls.append((model_id, kwargs))
        return self.value


class FakeDataset:
    calls: ClassVar[list[list[dict[str, object]]]] = []

    @classmethod
    def from_list(cls, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        cls.calls.append(rows)
        return rows


class FakeConfiguration:
    def __init__(self, **kwargs: object) -> None:
        self.values = kwargs


class FakeTorch:
    float32 = "float32-object"
    float16 = "float16-object"
    bfloat16 = "bfloat16-object"

    def __init__(self) -> None:
        self.seed: int | None = None
        self.cuda = SimpleNamespace(manual_seed_all=lambda seed: None)

    def manual_seed(self, seed: int) -> None:
        self.seed = seed


class FakeTrainer:
    instances: ClassVar[list[FakeTrainer]] = []
    save_full_weights = False
    training_error: RuntimeError | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.model = FakePeftModel(save_full_weights=self.save_full_weights)
        self.state = SimpleNamespace(log_history=[{"loss": 1.0}])
        self.resume_value: str | None = None
        self.__class__.instances.append(self)

    def train(self, *, resume_from_checkpoint: str | None) -> SimpleNamespace:
        self.resume_value = resume_from_checkpoint
        if self.training_error is not None:
            raise self.training_error
        return SimpleNamespace(metrics={"train_loss": 1.25, "epoch": 3})

    def evaluate(self) -> dict[str, float]:
        return {"eval_loss": 0.75}


def _config() -> TrainingConfig:
    value = load_training_config(CONFIG_PATH).model_dump(mode="json")
    data = value["data"]
    assert isinstance(data, dict)
    data["pilot_size_per_domain"] = 2
    data["train_size_per_domain"] = 1
    data["validation_size_per_domain"] = 1
    return TrainingConfig.model_validate(value)


def _hardware(device: str = "cpu", dtype: str = "float32") -> TrainingHardwareReport:
    return TrainingHardwareReport(
        ready=True,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        revision="a" * 40,
        python_version="3.11.9",
        platform_system="Windows",
        platform_machine="AMD64",
        package_versions={"torch": "test"},
        selected_device=device,  # type: ignore[arg-type]
        selected_dtype=dtype,  # type: ignore[arg-type]
        estimated_fit="cpu_debug_only" if device == "cpu" else "likely",
        cpu_debug_only=device == "cpu",
        cuda_available=device == "cuda",
        mps_built=device == "mps",
        mps_available=device == "mps",
        mps_fallback_enabled=False,
        system_ram_gb=16,
        model_cache_path="C:/cache/model",
        model_cached=True,
        artifact_root="C:/artifacts",
        artifact_root_writable=True,
    )


def _record(split: TrainingSplit, source_id: str) -> SFTTrainingRecord:
    return SFTTrainingRecord(
        source_id=source_id,
        domain=Domain.MATH,
        split=split,
        source_content_sha256="c" * 64,
        prompt=[
            ChatMessage(role="system", content="General prompt"),
            ChatMessage(role="user", content="What is 1 + 1?"),
        ],
        completion=[ChatMessage(role="assistant", content="Final answer: 2")],
        prompt_tokens=10,
        completion_tokens=4,
    )


def _modules(
    tokenizer: FakeTokenizer | None = None,
    model: FakeBaseModel | None = None,
) -> tuple[TrainingModules, FakeFactory, FakeFactory, FakeTorch]:
    tokenizer_factory = FakeFactory(tokenizer or FakeTokenizer())
    model_factory = FakeFactory(model or FakeBaseModel())
    torch_module = FakeTorch()
    modules = TrainingModules(
        torch=torch_module,
        auto_tokenizer=tokenizer_factory,
        auto_model=model_factory,
        dataset=FakeDataset,
        lora_config=FakeConfiguration,
        task_type=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
        sft_config=FakeConfiguration,
        sft_trainer=FakeTrainer,
        package_versions={
            "torch": "test",
            "transformers": "test",
            "datasets": "test",
            "peft": "test",
            "trl": "test",
            "accelerate": "test",
            "safetensors": "test",
        },
    )
    return modules, tokenizer_factory, model_factory, torch_module


@pytest.fixture(autouse=True)
def reset_fakes() -> None:
    FakeDataset.calls = []
    FakeTrainer.instances = []
    FakeTrainer.save_full_weights = False
    FakeTrainer.training_error = None


def test_loads_safe_model_and_builds_native_prompt_completion_trainer(tmp_path: Path) -> None:
    config = _config()
    modules, tokenizer_factory, model_factory, torch_module = _modules()
    backend = LoraTrainerBackend(config, _hardware(), modules, clock=iter([10.0, 12.5]).__next__)
    checkpoint = tmp_path / "checkpoint-1"

    result = backend.train(
        Domain.MATH,
        [_record(TrainingSplit.TRAIN, "train-1")],
        [_record(TrainingSplit.VALIDATION, "validation-1")],
        tmp_path,
        checkpoint,
    )

    common = {
        "revision": config.model.revision,
        "trust_remote_code": False,
        "local_files_only": False,
    }
    assert tokenizer_factory.calls == [(config.model.model_id, common)]
    assert model_factory.calls == [
        (
            config.model.model_id,
            {
                **common,
                "use_safetensors": True,
                "dtype": "float32-object",
                "low_cpu_mem_usage": True,
            },
        )
    ]
    assert backend.model.config.use_cache is False
    assert backend.model.device == "cpu"
    assert backend.model.checkpoint_kwargs == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }
    assert backend.tokenizer.pad_token == backend.tokenizer.eos_token
    assert torch_module.seed == 42

    trainer = FakeTrainer.instances[0]
    sft_values = trainer.kwargs["args"].values
    assert sft_values["max_length"] == 512
    assert sft_values["completion_only_loss"] is True
    assert sft_values["packing"] is False
    assert sft_values["load_best_model_at_end"] is True
    assert sft_values["metric_for_best_model"] == "eval_loss"
    assert sft_values["use_cpu"] is True
    assert sft_values["fp16"] is False
    assert sft_values["bf16"] is False
    lora_values = trainer.kwargs["peft_config"].values
    assert lora_values["r"] == 8
    assert lora_values["lora_alpha"] == 16
    assert lora_values["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert FakeDataset.calls[0][0]["prompt"][0]["role"] == "system"
    assert FakeDataset.calls[0][0]["completion"][0]["role"] == "assistant"
    assert trainer.resume_value == str(checkpoint)

    assert result.trainable_parameters == 16
    assert result.total_parameters == 1_016
    assert result.duration_seconds == pytest.approx(2.5)
    assert result.train_metrics["train_loss"] == pytest.approx(1.25)
    assert result.eval_metrics["eval_loss"] == pytest.approx(0.75)
    assert len(result.adapter_sha256) == 64
    assert (result.adapter_dir / "adapter_model.safetensors").is_file()
    metrics = json.loads((result.adapter_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation"]["eval_loss"] == pytest.approx(0.75)


def test_cuda_precision_flags_are_forwarded() -> None:
    modules, _tokenizer_factory, _model_factory, _torch_module = _modules()
    backend = LoraTrainerBackend(_config(), _hardware("cuda", "bfloat16"), modules)

    values = backend._sft_configuration(Path("work")).values

    assert values["bf16"] is True
    assert values["fp16"] is False
    assert values["use_cpu"] is False


def test_rejects_wrong_record_counts_before_constructing_trainer(tmp_path: Path) -> None:
    modules, _tokenizer_factory, _model_factory, _torch_module = _modules()
    backend = LoraTrainerBackend(_config(), _hardware(), modules)

    with pytest.raises(ValueError, match="math train has 0 rows; 1 required"):
        backend.train(
            Domain.MATH,
            [],
            [_record(TrainingSplit.VALIDATION, "validation-1")],
            tmp_path,
        )

    assert FakeTrainer.instances == []


def test_rejects_full_model_weights_in_adapter_output(tmp_path: Path) -> None:
    FakeTrainer.save_full_weights = True
    modules, _tokenizer_factory, _model_factory, _torch_module = _modules()
    backend = LoraTrainerBackend(_config(), _hardware(), modules)

    with pytest.raises(RuntimeError, match="full base-model weights"):
        backend.train(
            Domain.MATH,
            [_record(TrainingSplit.TRAIN, "train-1")],
            [_record(TrainingSplit.VALIDATION, "validation-1")],
            tmp_path,
        )


def test_wraps_out_of_memory_with_actionable_fallback(tmp_path: Path) -> None:
    FakeTrainer.training_error = RuntimeError("MPS backend out of memory")
    modules, _tokenizer_factory, _model_factory, _torch_module = _modules()
    backend = LoraTrainerBackend(_config(), _hardware(), modules)

    with pytest.raises(TrainingOutOfMemoryError, match="384-token"):
        backend.train(
            Domain.MATH,
            [_record(TrainingSplit.TRAIN, "train-1")],
            [_record(TrainingSplit.VALIDATION, "validation-1")],
            tmp_path,
        )


def test_lazy_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)

    with pytest.raises(TrainingDependencyError, match=r"requirements-training\.lock"):
        load_training_modules()
