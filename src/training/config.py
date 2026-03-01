"""
config.py — QLoRA finetuning configuration for Ministral 3B.

Optimized for:
- Single GPU (A100 40GB or 2x T4)
- 30-hour hackathon timeline
- Dramatic before/after improvement
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    # Base model — Ministral 3B
    model_id: str = "mistralai/Ministral-8B-Instruct-2410"
    
    # Output
    output_dir: str = "models/penumbra-ministral-8b"
    
    # Context length
    max_seq_length: int = 512


@dataclass  
class LoraConfig:
    # QLoRA parameters
    r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    
    target_modules: list = None
    
    def __post_init__(self):
        self.target_modules = ["q_proj", "v_proj"]
    
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    # Core training params
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    
    # Batch size — adjust based on GPU memory
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    
    # Optimizer — 8-bit for memory efficiency
    optim: str = "paged_adamw_8bit"
    
    # Precision
    bf16: bool = True
    tf32: bool = True
    
    # Scheduler
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 50
    logging_steps: int = 10
    
    # Best model
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Data
    dataloader_num_workers: int = 4
    
    # W&B
    report_to: str = "wandb"
    run_name: str = "penumbra-ministral-8b"


@dataclass
class QuantizationConfig:
    # 4-bit quantization for QLoRA
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


# Instantiate configs
model_config = ModelConfig()
lora_config = LoraConfig()
training_config = TrainingConfig()
quant_config = QuantizationConfig()

# Paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)