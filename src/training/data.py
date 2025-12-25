"""
Dataset loading and loss masking for Ventriloquist training.

File: training/data.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-24
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset

log = logging.getLogger(__name__)


def load_windows(path: Path) -> List[Dict]:
    """Load training windows from JSONL file."""
    windows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                windows.append(json.loads(line))
    log.info(f"Loaded {len(windows):,} windows from {path}")
    return windows


def extract_members_from_window(text: str) -> Tuple[str, ...]:
    """Extract sorted member tuple from window header."""
    first_line = text.split("\n")[0]
    header = json.loads(first_line)
    return tuple(sorted(header["members"]))


def train_eval_split(
    windows: List[Dict],
    holdout_per_chat: int = 2,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split windows into train and eval sets.

    Holds out the final N windows from each conversation (by participant set).
    """
    by_conversation: Dict[Tuple[str, ...], List[Dict]] = defaultdict(list)

    for window in windows:
        members = extract_members_from_window(window["text"])
        by_conversation[members].append(window)

    train_windows = []
    eval_windows = []

    for members, conv_windows in by_conversation.items():
        if len(conv_windows) <= holdout_per_chat:
            train_windows.extend(conv_windows)
        else:
            train_windows.extend(conv_windows[:-holdout_per_chat])
            eval_windows.extend(conv_windows[-holdout_per_chat:])

    log.info(f"Split: {len(train_windows):,} train, {len(eval_windows):,} eval")
    log.info(f"Conversations: {len(by_conversation)}")

    return train_windows, eval_windows


def find_last_message_start(text: str) -> int:
    """Find the character position where the final message begins."""
    last_newline_json = text.rfind('\n{"name":')
    if last_newline_json == -1:
        return 0
    return last_newline_json + 1


def create_dataset(
    windows: List[Dict],
    tokenizer,
    max_length: int = 4096,
) -> Dataset:
    """
    Create a HuggingFace Dataset with loss masking.

    Only computes loss on the final message of each window.
    """
    texts = [window["text"] for window in windows]

    def tokenize_with_masking(examples):
        results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for text in examples["text"]:
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Create labels with loss masking
            last_msg_char_pos = find_last_message_start(text)
            prefix_text = text[:last_msg_char_pos]
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            mask_until = len(prefix_tokens)

            labels = input_ids.copy()
            for i in range(min(mask_until, len(labels))):
                labels[i] = -100

            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            results["labels"].append(labels)

        return results

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(
        tokenize_with_masking,
        batched=True,
        remove_columns=["text"],
    )

    return dataset


def create_data_collator(tokenizer):
    """Create a data collator for variable-length sequences."""
    from transformers import DataCollatorForSeq2Seq

    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
