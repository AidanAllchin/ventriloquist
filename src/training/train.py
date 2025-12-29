"""
Main training script for Ventriloquist.

Fine-tunes Qwen3-8B-Base with LoRA using Unsloth for 2-5x speedup.

Usage:
    >>> python -m src.training.train
    >>> python -m src.training.train --lora_r 64 --learning_rate 1e-4

File: training/train.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-28
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

# Unsloth MUST be imported before transformers, peft, trl to apply optimizations
# See: https://github.com/unslothai/unsloth#installation
try:
    import unsloth  # type: ignore
except ImportError:
    pass  # Training extras not installed

import wandb
from transformers import TrainingArguments, TrainerCallback


class WandbMetricsCallback(TrainerCallback):
    """Explicitly log metrics to WandB since Unsloth may interfere with default logging."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and wandb.run is not None:
            # Filter out non-numeric values and log to wandb
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if metrics:
                wandb.log(metrics, step=state.global_step)

from .config import TrainingConfig
from .data import (
    create_dataset,
    create_data_collator,
    load_windows,
    train_eval_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> TrainingConfig:
    """Parse command line arguments and return config."""
    parser = argparse.ArgumentParser(description="Train Ventriloquist model")

    # Model
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)

    # Training
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)

    # Data
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit total windows for testing")

    # Logging & checkpoints
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Continue training from a saved adapter (e.g., checkpoints/ventriloquist/final)")

    args = parser.parse_args()

    # Start with defaults
    config = TrainingConfig()

    # Override with CLI args
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    # Ensure Path types after CLI overrides (since __post_init__ ran before overrides)
    if isinstance(config.output_dir, str):
        config.output_dir = Path(config.output_dir)
    if isinstance(config.data_path, str):
        config.data_path = Path(config.data_path)

    if args.no_wandb:
        config.report_to = "none"

    # Skip warmup for continued training (model already warmed up)
    if args.continue_from and args.warmup_ratio is None:
        config.warmup_ratio = 0.0
        log.info("Auto-setting warmup_ratio=0 for continued training")

    # Generate run name if not provided
    if config.run_name is None:
        config.run_name = f"ventriloquist-r{config.lora_r}-{datetime.now():%Y%m%d-%H%M}"

    return config


def load_model_and_tokenizer(config: TrainingConfig):
    """Load model and tokenizer using Unsloth."""
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError:
        raise ImportError(
            "Training requires unsloth. Install with: uv pip install 'ventriloquist[train]'"
        )

    log.info(f"Loading model: {config.base_model}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect (bf16 on supported GPUs)
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info(f"Model loaded with max_seq_length={config.max_seq_length}")

    return model, tokenizer


def apply_lora(model, config: TrainingConfig):
    """Apply LoRA adapters using Unsloth."""
    from unsloth import FastLanguageModel  # type: ignore

    log.info(f"Applying LoRA: r={config.lora_r}, alpha={config.lora_alpha}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=42,
    )

    return model


def main():
    """Main training function."""
    try:
        from trl.trainer.sft_trainer import SFTTrainer  # type: ignore
    except ImportError:
        raise ImportError(
            "Training requires trl. Install with: uv pip install 'ventriloquist[train]'"
        )

    config = parse_args()

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    if config.report_to == "wandb":
        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=config.to_dict(),
        )

    # Load model and tokenizer
    if config.continue_from:
        # Continue training from a saved adapter
        from unsloth import FastLanguageModel  # type: ignore

        log.info(f"Continuing training from: {config.continue_from}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.continue_from,
            max_seq_length=config.max_seq_length,
            dtype=None,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        FastLanguageModel.for_training(model)
    else:
        # Fresh training from base model
        model, tokenizer = load_model_and_tokenizer(config)
        model = apply_lora(model, config)

    # Load and split data
    log.info(f"Loading data from {config.data_path}")
    windows = load_windows(config.data_path)

    # Optional limit for testing
    if config.limit:
        log.info(f"Limiting to {config.limit} windows for testing")
        windows = windows[:config.limit]

    train_windows, eval_windows = train_eval_split(
        windows,
        holdout_per_chat=config.eval_holdout_per_chat,
    )

    # Create datasets (with caching for faster restarts)
    cache_dir = config.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_suffix = f"_limit{config.limit}" if config.limit else ""
    train_dataset = create_dataset(
        train_windows, tokenizer, config.max_seq_length,
        cache_file=str(cache_dir / f"train_tokenized{cache_suffix}.arrow"),
    )
    eval_dataset = create_dataset(
        eval_windows, tokenizer, config.max_seq_length,
        cache_file=str(cache_dir / f"eval_tokenized{cache_suffix}.arrow"),
    )

    # Shuffle training data so we don't train on contacts in sequence
    train_dataset = train_dataset.shuffle(seed=42)

    log.info(f"Train dataset: {len(train_dataset):,} windows")
    log.info(f"Eval dataset: {len(eval_dataset):,} windows")

    # Data collator
    data_collator = create_data_collator(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        run_name=config.run_name,
        # Batch
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Epochs
        num_train_epochs=config.num_epochs,
        # Optimizer
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        # Logging
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        # Saving
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=10,
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        prediction_loss_only=True,  # Only compute loss, don't return massive logits tensor
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Misc
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        # Unsloth optimizations
        optim="adamw_8bit",
    )

    # Use SFTTrainer for better compatibility with Unsloth
    callbacks = []
    if config.report_to == "wandb":
        callbacks.append(WandbMetricsCallback())

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    log.info("Starting training...")
    log.info(f"  Epochs: {config.num_epochs}")
    log.info(f"  Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.effective_batch_size}")
    log.info(f"  Steps per epoch: ~{len(train_dataset) // config.effective_batch_size:,}")
    log.info(f"  Learning rate: {config.learning_rate}")
    log.info(f"  LR scheduler: {config.lr_scheduler_type}")

    trainer.train(resume_from_checkpoint=True if config.resume else None)

    # Save final model
    final_path = config.output_dir / "final"
    log.info(f"Saving final model to {final_path}")
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Finish W&B
    if config.report_to == "wandb":
        wandb.finish()

    log.info("Training complete!")


if __name__ == "__main__":
    main()
