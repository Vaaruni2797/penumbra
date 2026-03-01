"""
train.py — QLoRA finetuning of Ministral 8B for Penumbra uncertainty maps.

Run: python src/training/train.py
"""

import os
import json
import torch
import wandb
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

from config import (
    model_config, lora_config, training_config,
    quant_config, DATA_DIR
)

load_dotenv()


def load_dataset_from_jsonl(path: Path) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_messages(example: dict) -> dict:
    """Format messages list into a single text string."""
    messages = example["messages"]
    text = ""
    for msg in messages:
        if msg["role"] == "system":
            text += f"<s>[INST] <<SYS>>\n{msg['content']}\n<</SYS>>\n\n"
        elif msg["role"] == "user":
            text += f"{msg['content']} [/INST] "
        elif msg["role"] == "assistant":
            text += f"{msg['content']}</s>"
    return {"text": text}


def setup_model_and_tokenizer():
    """Load model with 4-bit QLoRA quantization."""
    print(f"Loading {model_config.model_id}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def build_peft_config():
    return LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type
    )


def train():
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "penumbra"),
        name=training_config.run_name,
        config={
            "model": model_config.model_id,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "learning_rate": training_config.learning_rate,
            "epochs": training_config.num_train_epochs,
            "batch_size": training_config.per_device_train_batch_size,
            "grad_accum": training_config.gradient_accumulation_steps,
        }
    )

    print("=" * 50)
    print("FINETUNING PENUMBRA")
    print("=" * 50)

    # Load and format datasets
    print("\nLoading datasets...")
    train_dataset = load_dataset_from_jsonl(DATA_DIR / "train.jsonl")
    val_dataset = load_dataset_from_jsonl(DATA_DIR / "val.jsonl")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_dataset = train_dataset.map(format_messages)
    val_dataset = val_dataset.map(format_messages)

    # Load model and peft config
    model, tokenizer = setup_model_and_tokenizer()
    peft_config = build_peft_config()

    # SFTConfig — exact parameter names from docs
    sft_config = SFTConfig(
        # Output
        output_dir=model_config.output_dir,

        # Training
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=0.001,
        max_grad_norm=0.3,
        optim=training_config.optim,
        bf16=training_config.bf16,
        gradient_checkpointing=True,

        # Evaluation
        eval_strategy=training_config.evaluation_strategy,
        eval_steps=training_config.eval_steps,
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,

        # SFT specific
        dataset_text_field="text",
        max_length=model_config.max_seq_length,
        packing=False,

        # Logging
        report_to=training_config.report_to,
        run_name=training_config.run_name,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        args=sft_config,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving model to {model_config.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(model_config.output_dir)

    wandb.finish()
    print("\n✅ Penumbra training complete!")


if __name__ == "__main__":
    train()