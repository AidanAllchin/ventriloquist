"""
Inference utilities for Ventriloquist.

Model loading and device detection.

File: inference/utils.py
Author: Aidan Allchin
Created: 2025-12-26
Last Modified: 2025-12-26
"""

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


def get_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    adapter_path: Path,
    max_seq_length: int = 4096,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """
    Load trained model with LoRA adapter.

    Args:
        adapter_path: Path to saved LoRA adapter
        max_seq_length: Maximum sequence length
        load_in_8bit: Use 8-bit quantization (CUDA only)
        load_in_4bit: Use 4-bit quantization (CUDA only)

    Returns:
        (model, tokenizer)
    """
    from ..training.config import TrainingConfig

    config = TrainingConfig()
    base_model_name = config.base_model

    # For inference, use the non-unsloth model name
    if base_model_name.startswith("unsloth/"):
        base_model_name = base_model_name.replace("unsloth/", "Qwen/")

    device = get_device()
    log.info(f"Loading base model: {base_model_name}")
    log.info(f"Loading adapter from: {adapter_path}")
    log.info(f"Device: {device}")

    # Quantization only works on CUDA
    if device != "cuda" and (load_in_8bit or load_in_4bit):
        log.warning("Quantization requires CUDA, ignoring quantization flags")
        load_in_8bit = False
        load_in_4bit = False

    if load_in_8bit:
        log.info("Using 8-bit quantization")
    elif load_in_4bit:
        log.info("Using 4-bit quantization")

    # Determine dtype based on device
    if device == "cuda":
        dtype = torch.float16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # Move to device if not using device_map
    if device != "cuda":
        model = model.to(device)  # type: ignore

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
