"""
Training configuration for Ventriloquist.

File: training/config.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-28
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for LoRA fine-tuning with Unsloth."""

    # Model (use unsloth's optimized version)
    base_model: str = "unsloth/Qwen3-8B-Base"
    max_seq_length: int = 4096
    load_in_4bit: bool = False
    load_in_8bit: bool = False  # Good middle ground for VRAM vs quality

    # LoRA
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    # Unsloth auto-detects target modules, but we can specify if needed
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    eval_batch_size: int = 1  # Must be small: eval returns full logits (~2.4GB per sample)
    gradient_accumulation_steps: int = 16
    num_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"  # cosine annealing for better fine-tuning

    # Data
    data_path: Path = field(default_factory=lambda: Path("data/training_windows.jsonl"))
    output_dir: Path = field(default_factory=lambda: Path("checkpoints/ventriloquist"))
    eval_holdout_per_chat: int = 2
    limit: Optional[int] = None  # Limit windows for testing

    # Hardware - Unsloth handles precision automatically
    # use_gradient_checkpointing = "unsloth" for 30% less VRAM

    # Logging
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    report_to: str = "wandb"
    wandb_project: str = "ventriloquist"
    run_name: Optional[str] = None
    resume: bool = False  # Resume from latest checkpoint
    continue_from: Optional[str] = None  # Continue training from saved adapter

    def __post_init__(self):
        # Ensure paths are Path objects
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict:
        """Convert to dict for W&B logging."""
        return {
            "base_model": self.base_model,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "num_epochs": self.num_epochs,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
        }
