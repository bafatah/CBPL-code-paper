from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "你是一位火电厂脱硫系统优化专家。请分析当前系统状态，根据决策规则给出下一时刻最优泵配置，并提供详细推理过程。"
)
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = DEFAULT_REPO_ROOT / "data" / "data_c2.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "outputs" / "lora"
DEFAULT_MERGED_OUTPUT_DIR = DEFAULT_REPO_ROOT / "outputs" / "merged_16bit"


@dataclass(frozen=True)
class SFTScriptConfig:
    data_path: Path = DEFAULT_DATA_PATH
    model_name: str = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    cache_dir: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    merged_output_dir: Path | None = DEFAULT_MERGED_OUTPUT_DIR
    max_seq_length: int = 2748
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    lora_rank: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    per_device_train_batch_size: int = 20
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    num_train_epochs: int = 20
    learning_rate: float = 1e-4
    logging_steps: int = 1
    weight_decay: float = 0.001
    save_steps: int = 10
    save_total_limit: int = 1
    seed: int = 3407
    max_train_samples: int | None = None
    resume_from_checkpoint: str | None = None
    hf_token: str | None = None
    hf_endpoint: str | None = None


def conversation_from_record(
    record: dict[str, Any],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(record["input"])},
        {"role": "assistant", "content": str(record["output"])},
    ]


def build_text_dataset(
    *,
    tokenizer: Any,
    records: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, list[str]]:
    conversations = [
        conversation_from_record(record, system_prompt=system_prompt)
        for record in records
    ]
    texts = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text": list(texts)}


def load_records(data_path: Path, *, max_train_samples: int | None = None) -> list[dict[str, Any]]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {data_path} to contain a JSON list of records.")
    records = [record for record in payload if isinstance(record, dict)]
    return records[:max_train_samples] if max_train_samples is not None else records


def configure_environment(config: SFTScriptConfig) -> None:
    hf_token = config.hf_token or os.getenv("HF_TOKEN", "").strip()
    hf_endpoint = config.hf_endpoint or os.getenv("HF_ENDPOINT", "").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def build_model_and_tokenizer(config: SFTScriptConfig) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        cache_dir=str(config.cache_dir) if config.cache_dir is not None else None,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        full_finetuning=config.full_finetuning,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


def build_hf_train_dataset(config: SFTScriptConfig, tokenizer: Any) -> Any:
    from datasets import Dataset

    records = load_records(config.data_path, max_train_samples=config.max_train_samples)
    text_dataset = build_text_dataset(tokenizer=tokenizer, records=records)
    return Dataset.from_dict(text_dataset)


def build_trainer(config: SFTScriptConfig, model: Any, tokenizer: Any, train_dataset: Any) -> Any:
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=str(config.output_dir),
        dataset_text_field="text",
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        lr_scheduler_type="linear",
        seed=config.seed,
        report_to="none",
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=False,
    )
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=training_args,
    )


def save_outputs(config: SFTScriptConfig, model: Any, tokenizer: Any) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    if config.merged_output_dir is not None and hasattr(model, "save_pretrained_merged"):
        config.merged_output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(
            str(config.merged_output_dir),
            tokenizer,
            save_method="merged_16bit",
        )


def run_training(config: SFTScriptConfig) -> Any:
    configure_environment(config)
    model, tokenizer = build_model_and_tokenizer(config)
    train_dataset = build_hf_train_dataset(config, tokenizer)
    trainer = build_trainer(config, model, tokenizer, train_dataset)
    train_kwargs = {}
    if config.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = config.resume_from_checkpoint
    trainer_stats = trainer.train(**train_kwargs)
    save_outputs(config, model, tokenizer)
    return trainer_stats


def parse_args() -> SFTScriptConfig:
    parser = argparse.ArgumentParser(description="Train a Qwen3-8B SFT model for CBPL data.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-name", default="unsloth/Qwen3-8B-unsloth-bnb-4bit")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--merged-output-dir", type=Path, default=DEFAULT_MERGED_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=2748)
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--full-finetuning", action="store_true")
    parser.add_argument("--skip-merged-save", action="store_true")
    args = parser.parse_args()
    merged_output_dir = None if args.skip_merged_save else args.merged_output_dir
    return SFTScriptConfig(
        data_path=args.data_path,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        merged_output_dir=merged_output_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.load_in_8bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        resume_from_checkpoint=args.resume_from_checkpoint,
        hf_token=args.hf_token,
        hf_endpoint=args.hf_endpoint,
    )


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
