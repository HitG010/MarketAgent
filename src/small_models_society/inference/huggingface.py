"""Lazy Transformers backend for pinned Qwen causal language models."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from small_models_society.inference.config import InferenceConfig
from small_models_society.inference.contracts import GenerationOutput, GenerationRequest
from small_models_society.inference.hardware import HardwareReport


class InferenceDependencyError(RuntimeError):
    """Raised when the optional local inference stack is unavailable."""


class InferenceOutOfMemoryError(RuntimeError):
    """Raised when generation exhausts device memory."""


@dataclass(frozen=True)
class InferenceModules:
    torch: Any
    auto_tokenizer: Any
    auto_model: Any
    torch_version: str
    transformers_version: str


def load_inference_modules() -> InferenceModules:
    """Import optional ML packages only when a real backend is requested."""

    try:
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
        importlib.import_module("safetensors")
    except (ImportError, OSError) as error:
        raise InferenceDependencyError(
            "Local inference dependencies are unavailable. Install requirements-inference.lock."
        ) from error
    return InferenceModules(
        torch=torch_module,
        auto_tokenizer=transformers_module.AutoTokenizer,
        auto_model=transformers_module.AutoModelForCausalLM,
        torch_version=str(torch_module.__version__),
        transformers_version=str(transformers_module.__version__),
    )


class HuggingFaceBackend:
    """Load one pinned model and generate deterministic single-example outputs."""

    def __init__(
        self,
        config: InferenceConfig,
        hardware: HardwareReport,
        modules: InferenceModules | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not hardware.ready:
            raise ValueError("hardware report is not ready for inference")
        self.config = config
        self.hardware = hardware
        self.modules = modules or load_inference_modules()
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
        self.model = self.modules.auto_model.from_pretrained(
            config.model.model_id,
            **model_options,
            use_safetensors=config.model.use_safetensors,
            dtype=dtype,
        )
        self.model.to(hardware.selected_device)
        self.model.eval()
        self.modules.torch.manual_seed(config.generation.seed)
        if hardware.selected_device == "cuda":
            self.modules.torch.cuda.manual_seed_all(config.generation.seed)

    @staticmethod
    def _token_count(token_ids: Any) -> int:
        return int(token_ids.shape[-1])

    def _synchronize(self) -> None:
        if self.hardware.selected_device == "cuda":
            self.modules.torch.cuda.synchronize()
        elif self.hardware.selected_device == "mps":
            self.modules.torch.mps.synchronize()

    def _activate_request(
        self,
        request: GenerationRequest,
    ) -> AbstractContextManager[None]:
        if request.adapter is not None:
            raise ValueError("the base Hugging Face backend does not support adapters")
        return nullcontext()

    def _request_metadata(self, request: GenerationRequest) -> dict[str, object]:
        del request
        return {}

    @staticmethod
    def _is_out_of_memory(error: RuntimeError) -> bool:
        return "out of memory" in str(error).casefold()

    def _encode_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return dict(
            self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
        )

    def _truncate_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        user_indexes = [
            index for index, message in enumerate(messages) if message["role"] == "user"
        ]
        if not user_indexes:
            raise ValueError("generation request has no user message to truncate")
        user_index = user_indexes[-1]
        user_content = str(messages[user_index]["content"])
        low = 0
        high = len(user_content)
        best_encoding: dict[str, Any] | None = None
        while low <= high:
            midpoint = (low + high) // 2
            candidate_messages = [dict(message) for message in messages]
            candidate_messages[user_index]["content"] = user_content[:midpoint]
            candidate_encoding = self._encode_messages(candidate_messages)
            if self._token_count(candidate_encoding["input_ids"]) <= (
                self.config.generation.max_input_tokens
            ):
                best_encoding = candidate_encoding
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best_encoding is None:
            raise ValueError(
                "max_input_tokens is too small for the system prompt and chat control tokens"
            )
        return best_encoding

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        messages = [message.model_dump(mode="json") for message in request.messages]
        encoded = self._encode_messages(messages)
        original_prompt_tokens = self._token_count(encoded["input_ids"])
        input_truncated = original_prompt_tokens > self.config.generation.max_input_tokens
        if input_truncated:
            encoded = self._truncate_messages(messages)
        encoded = {
            name: tensor.to(self.hardware.selected_device) for name, tensor in encoded.items()
        }
        prompt_tokens = self._token_count(encoded["input_ids"])
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        self._synchronize()
        started = self._clock()
        try:
            with self._activate_request(request), self.modules.torch.inference_mode():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
            self._synchronize()
        except RuntimeError as error:
            if self._is_out_of_memory(error):
                raise InferenceOutOfMemoryError(
                    "Model generation ran out of memory. Run `sms inference doctor` "
                    "and reduce input/output token limits or use a machine with more memory."
                ) from error
            raise
        latency_ms = (self._clock() - started) * 1000
        generated_ids = output_ids[0][prompt_tokens:]
        completion_tokens = self._token_count(generated_ids)
        generated_values = list(generated_ids.tolist())
        if generated_values and generated_values[-1] == eos_token_id:
            stop_reason = "eos"
        elif completion_tokens >= request.max_new_tokens:
            stop_reason = "max_tokens"
        else:
            stop_reason = "other"
        text = str(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
        metadata = {
            "model_revision": self.config.model.revision,
            "device": self.hardware.selected_device,
            "dtype": self.hardware.selected_dtype,
            "profile": request.profile,
            "config_fingerprint": self.config.fingerprint(),
            "input_truncated": input_truncated,
            "original_prompt_tokens": original_prompt_tokens,
            "stop_reason": stop_reason,
            "torch_version": self.modules.torch_version,
            "transformers_version": self.modules.transformers_version,
            **self._request_metadata(request),
        }
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            metadata=metadata,
        )
