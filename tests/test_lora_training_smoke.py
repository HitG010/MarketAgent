from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from small_models_society.inference.contracts import GenerationOutput

pytestmark = pytest.mark.training


def test_tiny_qwen_lora_train_save_reload_switch_and_generate(tmp_path: Path) -> None:
    if os.getenv("SMS_RUN_TRAINING_TESTS") != "1":
        pytest.skip("set SMS_RUN_TRAINING_TESTS=1 to run the LoRA training smoke test")

    torch = importlib.import_module("torch")
    Dataset = importlib.import_module("datasets").Dataset
    peft = importlib.import_module("peft")
    LoraConfig = peft.LoraConfig
    PeftModel = peft.PeftModel
    TaskType = peft.TaskType
    Tokenizer = importlib.import_module("tokenizers").Tokenizer
    WordLevel = importlib.import_module("tokenizers.models").WordLevel
    Whitespace = importlib.import_module("tokenizers.pre_tokenizers").Whitespace
    transformers = importlib.import_module("transformers")
    PreTrainedTokenizerFast = transformers.PreTrainedTokenizerFast
    Qwen2Config = transformers.Qwen2Config
    Qwen2ForCausalLM = transformers.Qwen2ForCausalLM
    trl = importlib.import_module("trl")
    SFTConfig = trl.SFTConfig
    SFTTrainer = trl.SFTTrainer

    torch.manual_seed(42)
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "system": 4,
        "user": 5,
        "assistant": 6,
        "You": 7,
        "are": 8,
        "careful": 9,
        "What": 10,
        "is": 11,
        "one": 12,
        "plus": 13,
        "two": 14,
        "Final": 15,
        "answer": 16,
        "three": 17,
    }
    tokenizer_backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] + ' ' + message['content'] + ' ' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        chat_template=chat_template,
    )
    model_config = Qwen2Config(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(model_config)
    rows = [
        {
            "prompt": [
                {"role": "system", "content": "You are careful"},
                {"role": "user", "content": "What is one plus two"},
            ],
            "completion": [{"role": "assistant", "content": "Final answer three"}],
        }
    ]
    dataset = Dataset.from_list(rows)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(tmp_path / "checkpoints"),
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            use_cpu=True,
            max_length=64,
            completion_only_loss=True,
            packing=False,
            eos_token="<eos>",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )

    train_result = trainer.train()

    assert train_result.metrics["train_loss"] >= 0
    trainable = sum(
        parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    assert 0 < trainable < total

    adapter_dir = tmp_path / "adapter"
    trainer.model.save_pretrained(
        adapter_dir,
        safe_serialization=True,
        save_embedding_layers=False,
    )
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    assert (adapter_dir / "adapter_config.json").is_file()
    assert not (adapter_dir / "model.safetensors").exists()
    assert not (adapter_dir / "pytorch_model.bin").exists()

    reloaded = PeftModel.from_pretrained(
        Qwen2ForCausalLM(model_config),
        adapter_dir,
        adapter_name="math",
        is_trainable=False,
    )
    reloaded.load_adapter(adapter_dir, adapter_name="copy", is_trainable=False)
    reloaded.set_adapter("copy")
    encoded = tokenizer(
        "system You are careful user What is one plus two assistant",
        return_tensors="pt",
    )
    encoded.pop("token_type_ids", None)
    reloaded.eval()
    with torch.inference_mode():
        adapted_ids = reloaded.generate(
            **encoded,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with reloaded.disable_adapter():
            base_ids = reloaded.generate(
                **encoded,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    adapted_completion = adapted_ids[0, prompt_tokens:]
    base_completion = base_ids[0, prompt_tokens:]
    assert adapted_completion.numel() > 0
    assert base_completion.numel() > 0
    response = " ".join(str(int(token_id)) for token_id in adapted_completion)
    output = GenerationOutput(
        text=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=int(adapted_completion.numel()),
        latency_ms=0,
        metadata={"adapter": "copy"},
    )
    assert output.text.strip()
    assert output.completion_tokens > 0
