"""
Evaluation utilities for Ventriloquist.

Compute metrics on holdout set, per-contact analysis, and window comparison.

Usage:
    # Standard evaluation
    >>> python -m src.training.evaluate --adapter_path checkpoints/ventriloquist/final

    # Compare specific window (index into eval set)
    >>> python -m src.training.evaluate --compare 5
    >>> python -m src.training.evaluate --compare 5 --target "John Doe"
    >>> python -m src.training.evaluate --compare 5 --with_base

File: training/evaluate.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-24
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

from .data import load_windows, train_eval_split, create_dataset
from .inference import generate_response, build_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

VALID_DELTAS = {"<1m>", "<5m>", "<1h>", "<12h>", "<1d>", "1d+"}
BASE_MODEL = "unsloth/Qwen3-8B-Base"


def load_model(adapter_path: Path, max_seq_length: int = 4096):
    """Load fine-tuned model with LoRA adapter."""
    log.info(f"Loading fine-tuned model from: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_base_model(max_seq_length: int = 4096):
    """Load the original base model without any fine-tuning."""
    log.info(f"Loading base model: {BASE_MODEL}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def parse_window(window: Dict) -> Optional[Dict]:
    """Parse a window into header, context messages, and final message."""
    text = window["text"]
    lines = text.strip().split("\n")

    if len(lines) < 2:
        return None

    try:
        header = json.loads(lines[0])
        messages = [json.loads(line) for line in lines[1:-1]]
        final_msg = json.loads(lines[-1])

        return {
            "header": header,
            "context": messages,
            "expected": final_msg,
        }
    except json.JSONDecodeError:
        return None


def compare_window(
    window: Dict,
    model,
    tokenizer,
    target_name: Optional[str] = None,
    base_model=None,
    base_tokenizer=None,
) -> Dict:
    """
    Compare expected output vs generated for a specific window.

    Args:
        window: Training window dict
        model: Fine-tuned model
        tokenizer: Tokenizer
        target_name: Override who to generate for (default: use expected sender)
        base_model: Optional base model for comparison
        base_tokenizer: Tokenizer for base model

    Returns:
        Dict with comparison results
    """
    parsed = parse_window(window)
    if not parsed:
        return {"error": "Failed to parse window"}

    header = parsed["header"]
    context = parsed["context"]
    expected = parsed["expected"]

    # Use expected sender or override
    actual_target = target_name if target_name else expected["name"]

    # Generate with fine-tuned model
    finetuned_response = generate_response(
        model,
        tokenizer,
        context,
        target_name=actual_target,
        chat_type=header["type"],
        members=header["members"],
        max_new_tokens=150,
        temperature=0.7,
    )

    result = {
        "header": header,
        "context_count": len(context),
        "last_context": context[-1] if context else None,
        "expected": expected,
        "target_used": actual_target,
        "target_matches_expected": actual_target == expected["name"],
        "finetuned": {
            "delta": finetuned_response.delta if finetuned_response else None,
            "content": finetuned_response.content if finetuned_response else None,
            "valid": finetuned_response is not None,
        },
    }

    # Generate with base model if provided
    if base_model is not None:
        base_response = generate_response(
            base_model,
            base_tokenizer,
            context,
            target_name=actual_target,
            chat_type=header["type"],
            members=header["members"],
            max_new_tokens=150,
            temperature=0.7,
        )
        result["base"] = {
            "delta": base_response.delta if base_response else None,
            "content": base_response.content if base_response else None,
            "valid": base_response is not None,
        }

    return result


def print_comparison(result: Dict):
    """Pretty-print a comparison result."""
    print("\n" + "=" * 60)
    print(f"Chat: {result['header']['type']} with {result['header']['members']}")
    print(f"Context: {result['context_count']} messages")

    if result.get("last_context"):
        last = result["last_context"]
        print(f"Last message: {last['name']}: {last['content'][:80]}...")

    print("-" * 60)
    print(f"Target: {result['target_used']}", end="")
    if not result["target_matches_expected"]:
        print(f" (expected was: {result['expected']['name']})")
    else:
        print()

    print("-" * 60)
    exp = result["expected"]
    print(f"EXPECTED [{exp['delta']}]: {exp['content']}")

    print("-" * 60)
    ft = result["finetuned"]
    if ft["valid"]:
        print(f"FINETUNED [{ft['delta']}]: {ft['content']}")
    else:
        print("FINETUNED: [generation failed]")

    if "base" in result:
        print("-" * 60)
        base = result["base"]
        if base["valid"]:
            print(f"BASE [{base['delta']}]: {base['content']}")
        else:
            print("BASE: [generation failed]")

    print("=" * 60)


def compute_perplexity(
    model,
    tokenizer,
    eval_dataset,
    batch_size: int = 4,
) -> Dict[str, float]:
    """Compute perplexity on eval set (only on final message tokens)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Computing perplexity"):
            batch_end = min(i + batch_size, len(eval_dataset))
            batch_items = [eval_dataset[j] for j in range(i, batch_end)]

            max_len = max(len(item["input_ids"]) for item in batch_items)

            input_ids = torch.zeros(len(batch_items), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(batch_items), max_len, dtype=torch.long)
            labels = torch.full((len(batch_items), max_len), -100, dtype=torch.long)

            for j, item in enumerate(batch_items):
                seq_len = len(item["input_ids"])
                input_ids[j, :seq_len] = torch.tensor(item["input_ids"])
                attention_mask[j, :seq_len] = torch.tensor(item["attention_mask"])
                labels[j, :seq_len] = torch.tensor(item["labels"])

            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
    }


def evaluate_generation_quality(
    model,
    tokenizer,
    eval_windows: List[Dict],
    num_samples: int = 50,
) -> Dict[str, float]:
    """Evaluate generation quality: JSON validity, delta validity, content non-empty."""
    import random

    samples = random.sample(eval_windows, min(num_samples, len(eval_windows)))

    json_valid = 0
    delta_valid = 0
    content_nonempty = 0

    for window in tqdm(samples, desc="Evaluating generations"):
        parsed = parse_window(window)
        if not parsed or not parsed["context"]:
            continue

        response = generate_response(
            model,
            tokenizer,
            parsed["context"],
            target_name=parsed["expected"]["name"],
            chat_type=parsed["header"]["type"],
            members=parsed["header"]["members"],
            max_new_tokens=100,
            temperature=0.7,
        )

        if response is not None:
            json_valid += 1
            if response.delta in VALID_DELTAS:
                delta_valid += 1
            if response.content.strip():
                content_nonempty += 1

    n = len(samples)
    return {
        "json_valid_rate": json_valid / n if n > 0 else 0,
        "delta_valid_rate": delta_valid / n if n > 0 else 0,
        "content_nonempty_rate": content_nonempty / n if n > 0 else 0,
        "samples_evaluated": n,
    }


def per_contact_analysis(
    model,
    tokenizer,
    eval_windows: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by contact."""
    by_contact: Dict[str, List[Dict]] = defaultdict(list)

    for window in eval_windows:
        parsed = parse_window(window)
        if parsed:
            contact = parsed["expected"]["name"]
            by_contact[contact].append(window)

    results = {}

    for contact, windows in by_contact.items():
        log.info(f"Evaluating {contact}: {len(windows)} windows")
        sample = windows[:20] if len(windows) > 20 else windows

        metrics = evaluate_generation_quality(
            model, tokenizer, sample, num_samples=len(sample)
        )
        metrics["num_windows"] = len(windows)
        results[contact] = metrics

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ventriloquist model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/ventriloquist/final",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/training_windows.jsonl",
    )
    parser.add_argument("--skip_perplexity", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--per_contact", action="store_true")

    # Window comparison options
    parser.add_argument(
        "--compare",
        type=int,
        help="Compare specific window by index (into eval set)",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Override target name for comparison (default: use expected)",
    )
    parser.add_argument(
        "--with_base",
        action="store_true",
        help="Also generate with base model for comparison",
    )

    args = parser.parse_args()

    # Load data
    log.info("Loading evaluation data...")
    windows = load_windows(Path(args.data_path))
    _, eval_windows = train_eval_split(windows, holdout_per_chat=2)
    log.info(f"Eval set: {len(eval_windows)} windows")

    # Window comparison mode
    if args.compare is not None:
        if args.compare < 0 or args.compare >= len(eval_windows):
            log.error(f"Invalid window index: {args.compare} (valid: 0-{len(eval_windows)-1})")
            return

        log.info("Loading fine-tuned model...")
        model, tokenizer = load_model(Path(args.adapter_path))

        base_model, base_tokenizer = None, None
        if args.with_base:
            base_model, base_tokenizer = load_base_model()

        result = compare_window(
            eval_windows[args.compare],
            model,
            tokenizer,
            target_name=args.target,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
        )
        print_comparison(result)
        return

    # Standard evaluation mode
    log.info("Loading fine-tuned model...")
    model, tokenizer = load_model(Path(args.adapter_path))

    results = {}

    if not args.skip_perplexity:
        log.info("\n=== Computing Perplexity ===")
        eval_dataset = create_dataset(eval_windows, tokenizer, max_length=4096)
        ppl_results = compute_perplexity(model, tokenizer, eval_dataset)
        results["perplexity"] = ppl_results
        log.info(f"Perplexity: {ppl_results['perplexity']:.2f}")
        log.info(f"Avg Loss: {ppl_results['avg_loss']:.4f}")

    if not args.skip_generation:
        log.info("\n=== Evaluating Generation Quality ===")
        gen_results = evaluate_generation_quality(model, tokenizer, eval_windows, num_samples=50)
        results["generation"] = gen_results
        log.info(f"JSON Valid: {gen_results['json_valid_rate']:.1%}")
        log.info(f"Delta Valid: {gen_results['delta_valid_rate']:.1%}")
        log.info(f"Content Non-empty: {gen_results['content_nonempty_rate']:.1%}")

    if args.per_contact:
        log.info("\n=== Per-Contact Analysis ===")
        contact_results = per_contact_analysis(model, tokenizer, eval_windows)
        results["per_contact"] = contact_results

        for contact, metrics in sorted(contact_results.items()):
            log.info(f"\n{contact}:")
            log.info(f"  Windows: {metrics['num_windows']}")
            log.info(f"  JSON Valid: {metrics['json_valid_rate']:.1%}")
            log.info(f"  Delta Valid: {metrics['delta_valid_rate']:.1%}")

    output_path = Path(args.adapter_path) / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
