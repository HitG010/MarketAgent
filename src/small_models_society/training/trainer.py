"""Lazy PEFT/TRL backend for one LoRA specialist training process."""

from __future__ import annotations

import importlib
import numbers
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from small_models_society.data.prepare import canonical_json, sha256_bytes
from small_models_society.schemas import Domain
from small_models_society.training.config import TrainingConfig
from small_models_society.training.contracts import SFTTrainingRecord, TrainingSplit
from small_models_society.training.hardware import TrainingHardwareReport


class TrainingDependencyError(RuntimeError):
    """Raised when the optional LoRA training stack is unavailable."""


class TrainingOutOfMemoryError(RuntimeError):
    """Raised when a training step exhausts accelerator memory."""


@dataclass(frozen=True)
class TrainingModules:
    torch: Any
    auto_tokenizer: Any
    auto_model: Any
    dataset: Any
    lora_config: Any
    task_type: Any
    sft_config: Any
    sft_trainer: Any
    package_versions: dict[str, str]


@dataclass(frozen=True)
class TrainerBackendResult:
    specialist: Domain
    adapter_dir: Path
    adapter_sha256: str
    train_metrics: dict[str, object]
    eval_metrics: dict[str, object]
    trainable_parameters: int
    total_parameters: int
    duration_seconds: float
    resumed_from_checkpoint: str | None
    package_versions: dict[str, str]


def load_training_modules() -> TrainingModules:
    """Import the pinned optional training stack only when a real run starts."""

    try:
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
        datasets_module = importlib.import_module("datasets")
        peft_module = importlib.import_module("peft")
        trl_module = importlib.import_module("trl")
        accelerate_module = importlib.import_module("accelerate")
        safetensors_module = importlib.import_module("safetensors")
    except (ImportError, OSError) as error:
        raise TrainingDependencyError(
            "LoRA training dependencies are unavailable. Install requirements-training.lock."
        ) from error
    return TrainingModules(
        torch=torch_module,
        auto_tokenizer=transformers_module.AutoTokenizer,
        auto_model=transformers_module.AutoModelForCausalLM,
        dataset=datasets_module.Dataset,
        lora_config=peft_module.LoraConfig,
        task_type=peft_module.TaskType,
        sft_config=trl_module.SFTConfig,
        sft_trainer=trl_module.SFTTrainer,
        package_versions={
            "torch": str(torch_module.__version__),
            "transformers": str(transformers_module.__version__),
            "datasets": str(datasets_module.__version__),
            "peft": str(peft_module.__version__),
            "trl": str(trl_module.__version__),
            "accelerate": str(accelerate_module.__version__),
            "safetensors": str(safetensors_module.__version__),
        },
    )


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _metrics(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_value(item) for key, item in value.items()}


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _parameter_counts(model: Any) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if bool(parameter.requires_grad):
            trainable += count
    if total <= 0:
        raise RuntimeError("trainer model exposes no parameters")
    if trainable <= 0 or trainable >= total:
        raise RuntimeError(
            "LoRA parameter isolation failed: expected a nonzero strict subset "
            "of trainable parameters"
        )
    return trainable, total


def _adapter_sha256(adapter_dir: Path) -> str:
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not weights_path.is_file() or weights_path.stat().st_size <= 0:
        raise RuntimeError("trainer did not save adapter_model.safetensors")
    if not config_path.is_file():
        raise RuntimeError("trainer did not save adapter_config.json")
    prohibited = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    unexpected = [name for name in prohibited if (adapter_dir / name).exists()]
    unexpected.extend(path.name for path in adapter_dir.glob("model-*.safetensors"))
    if unexpected:
        raise RuntimeError(f"adapter output contains full base-model weights: {sorted(unexpected)}")
    return sha256_bytes(weights_path.read_bytes())


def _dataset_rows(records: Iterable[SFTTrainingRecord]) -> list[dict[str, object]]:
    return [
        {
            "prompt": [message.model_dump(mode="json") for message in record.prompt],
            "completion": [message.model_dump(mode="json") for message in record.completion],
        }
        for record in records
    ]


class LoraTrainerBackend:
    """Own one pinned base model and train one domain adapter."""

    def __init__(
        self,
        config: TrainingConfig,
        hardware: TrainingHardwareReport,
        modules: TrainingModules | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not hardware.ready:
            raise ValueError("hardware report is not ready for training")
        self.config = config
        self.hardware = hardware
        self.modules = modules or load_training_modules()
        self._clock = clock
        dtype = {
            "float32": self.modules.torch.float32,
            "float16": self.modules.torch.float16,
            "bfloat16": self.modules.torch.bfloat16,
        }[hardware.selected_dtype]
        model_options = {
            "revision": config.model.revision,
            "trust_remote_code": config.model.trust_remote_code,
            "local_files_only": config.model.local_files_only,
        }
        self.tokenizer = self.modules.auto_tokenizer.from_pretrained(
            config.model.model_id,
            **model_options,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.modules.auto_model.from_pretrained(
            config.model.model_id,
            **model_options,
            use_safetensors=config.model.use_safetensors,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.to(hardware.selected_device)
        self.modules.torch.manual_seed(config.data.seed)
        if hardware.selected_device == "cuda":
            self.modules.torch.cuda.manual_seed_all(config.data.seed)

    def _lora_configuration(self) -> Any:
        return self.modules.lora_config(
            task_type=self.modules.task_type.CAUSAL_LM,
            base_model_name_or_path=self.config.model.model_id,
            revision=self.config.model.revision,
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            target_modules=[module.value for module in self.config.lora.target_modules],
            init_lora_weights=self.config.lora.init_lora_weights,
        )

    def _sft_configuration(self, work_dir: Path) -> Any:
        sft = self.config.sft
        use_cuda_half = self.hardware.selected_device == "cuda"
        return self.modules.sft_config(
            output_dir=str(work_dir / "checkpoints"),
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            eval_strategy=sft.eval_strategy,
            per_device_train_batch_size=sft.per_device_train_batch_size,
            per_device_eval_batch_size=sft.per_device_eval_batch_size,
            gradient_accumulation_steps=sft.gradient_accumulation_steps,
            learning_rate=sft.learning_rate,
            num_train_epochs=sft.num_train_epochs,
            lr_scheduler_type=sft.lr_scheduler_type,
            warmup_ratio=sft.warmup_ratio,
            logging_strategy="steps",
            logging_steps=sft.logging_steps,
            save_strategy=sft.save_strategy,
            save_total_limit=sft.save_total_limit,
            save_safetensors=self.config.output.save_safetensors,
            load_best_model_at_end=sft.load_best_model_at_end,
            metric_for_best_model=sft.metric_for_best_model,
            greater_is_better=sft.greater_is_better,
            seed=self.config.data.seed,
            data_seed=self.config.data.seed,
            use_cpu=self.hardware.selected_device == "cpu",
            bf16=use_cuda_half and self.hardware.selected_dtype == "bfloat16",
            fp16=use_cuda_half and self.hardware.selected_dtype == "float16",
            gradient_checkpointing=sft.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to=sft.report_to,
            max_length=self.config.data.max_length,
            completion_only_loss=sft.completion_only_loss,
            packing=sft.packing,
            eos_token="<|im_end|>",
            dataset_num_proc=None,
        )

    def _validate_records(
        self,
        specialist: Domain,
        train_records: list[SFTTrainingRecord],
        validation_records: list[SFTTrainingRecord],
    ) -> None:
        expected = (
            (TrainingSplit.TRAIN, train_records, self.config.data.train_size_per_domain),
            (
                TrainingSplit.VALIDATION,
                validation_records,
                self.config.data.validation_size_per_domain,
            ),
        )
        for split, records, expected_count in expected:
            if len(records) != expected_count:
                raise ValueError(
                    f"{specialist.value} {split.value} has {len(records)} rows; "
                    f"{expected_count} required"
                )
            if any(
                record.domain is not specialist or record.split is not split for record in records
            ):
                raise ValueError(
                    f"{specialist.value} {split.value} records contain another domain or split"
                )
        train_ids = {record.source_id for record in train_records}
        if not train_ids.isdisjoint(record.source_id for record in validation_records):
            raise ValueError("specialist train and validation source IDs overlap")

    def train(
        self,
        specialist: Domain,
        train_records: list[SFTTrainingRecord],
        validation_records: list[SFTTrainingRecord],
        work_dir: Path,
        resume_from_checkpoint: Path | None = None,
    ) -> TrainerBackendResult:
        """Train and save exactly one adapter in an isolated work directory."""

        self._validate_records(specialist, train_records, validation_records)
        work_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = self.modules.dataset.from_list(_dataset_rows(train_records))
        validation_dataset = self.modules.dataset.from_list(_dataset_rows(validation_records))
        trainer = self.modules.sft_trainer(
            model=self.model,
            args=self._sft_configuration(work_dir),
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.tokenizer,
            peft_config=self._lora_configuration(),
        )
        trainable_parameters, total_parameters = _parameter_counts(trainer.model)
        started = self._clock()
        resume_value = str(resume_from_checkpoint) if resume_from_checkpoint else None
        try:
            train_result = trainer.train(resume_from_checkpoint=resume_value)
            eval_result = trainer.evaluate()
        except RuntimeError as error:
            if "out of memory" in str(error).casefold():
                raise TrainingOutOfMemoryError(
                    "LoRA training ran out of memory. Use the explicit 384-token "
                    "configuration or a machine with more accelerator memory."
                ) from error
            raise
        duration_seconds = self._clock() - started
        adapter_dir = work_dir / "adapter"
        trainer.model.save_pretrained(
            str(adapter_dir),
            safe_serialization=self.config.output.save_safetensors,
        )
        adapter_sha256 = _adapter_sha256(adapter_dir)
        train_metrics = _metrics(cast(Mapping[str, object], train_result.metrics))
        eval_metrics = _metrics(cast(Mapping[str, object], eval_result))
        _write_json(
            adapter_dir / "metrics.json",
            {"train": train_metrics, "validation": eval_metrics},
        )
        log_history = _json_value(getattr(trainer.state, "log_history", []))
        _write_json(adapter_dir / "log_history.json", log_history)
        return TrainerBackendResult(
            specialist=specialist,
            adapter_dir=adapter_dir,
            adapter_sha256=adapter_sha256,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            trainable_parameters=trainable_parameters,
            total_parameters=total_parameters,
            duration_seconds=duration_seconds,
            resumed_from_checkpoint=resume_value,
            package_versions=self.modules.package_versions,
        )
