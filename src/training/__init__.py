"""
Training module for Ventriloquist.

Fine-tunes Qwen3-8B-Base with LoRA using Unsloth.
"""

from .config import TrainingConfig
from .data import (
    create_dataset,
    load_windows,
    train_eval_split,
    create_data_collator,
)

__all__ = [
    "TrainingConfig",
    "create_dataset",
    "load_windows",
    "train_eval_split",
    "create_data_collator",
]
