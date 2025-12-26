"""
Inference utilities for Ventriloquist.

Load trained model and generate message completions interactively.

Usage:
    >>> python -m src.training.inference
    >>> python -m src.training.inference --adapter_path checkpoints/my_model

File: training/inference.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-25
"""

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from rich.text import Text
from unsloth import FastLanguageModel

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)


@dataclass
class GeneratedMessage:
    """A generated message with parsed fields."""

    name: str
    delta: str
    content: str
    raw: str  # Original generated text


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
        load_in_8bit: Use 8-bit quantization (for lower VRAM)
        load_in_4bit: Use 4-bit quantization (for even lower VRAM)

    Returns:
        (model, tokenizer)
    """
    log.info(f"Loading model from: {adapter_path}")
    if load_in_8bit:
        log.info("Using 8-bit quantization")
    elif load_in_4bit:
        log.info("Using 4-bit quantization")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def build_header(
    chat_type: str,
    members: List[str],
    start_date: Optional[str] = None,
) -> str:
    """
    Build the window header JSON.

    Args:
        chat_type: "dm" or "group"
        members: List of participant names
        start_date: Optional date string (defaults to today)

    Returns:
        JSON header string
    """
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    header = {
        "type": chat_type,
        "members": sorted(members),
        "start": start_date,
    }
    return json.dumps(header, ensure_ascii=False)


def build_prompt(
    context_messages: List[Dict[str, str]],
    target_name: str,
    chat_type: str = "dm",
    members: Optional[List[str]] = None,
    start_date: Optional[str] = None,
) -> str:
    """
    Build the full prompt for generation.

    Args:
        context_messages: List of {"name": ..., "delta": ..., "content": ...}
        target_name: Who to generate for
        chat_type: "dm" or "group"
        members: Participant names (inferred from messages if not provided)
        start_date: Window start date

    Returns:
        Full prompt string ending with target prefill
    """
    if members is None:
        members = list({msg["name"] for msg in context_messages})
        if target_name not in members:
            members.append(target_name)

    header = build_header(chat_type, members, start_date)

    message_lines = [
        json.dumps(msg, ensure_ascii=False)
        for msg in context_messages
    ]

    prefill = f'{{"name": "{target_name}", "'

    parts = [header] + message_lines + [prefill]
    return "\n".join(parts)


def parse_generated(raw_text: str, target_name: str) -> Optional[GeneratedMessage]:
    """
    Parse the generated text into a structured message.

    Args:
        raw_text: Raw model output (starting after prefill)
        target_name: Expected sender name

    Returns:
        GeneratedMessage if parsing succeeds, None otherwise
    """
    try:
        full_json = f'{{"name": "{target_name}", "{raw_text}'

        brace_count = 0
        end_pos = 0

        for i, char in enumerate(full_json):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break

        if end_pos == 0:
            full_json = full_json.rstrip() + "}"
            end_pos = len(full_json)

        json_str = full_json[:end_pos]
        parsed = json.loads(json_str)

        return GeneratedMessage(
            name=parsed.get("name", target_name),
            delta=parsed.get("delta", ""),
            content=parsed.get("content", ""),
            raw=raw_text,
        )

    except json.JSONDecodeError:
        log.warning(f"Failed to parse generated JSON: {raw_text[:100]}...")
        return None


def generate_response(
    model,
    tokenizer,
    context_messages: List[Dict[str, str]],
    target_name: str,
    chat_type: str = "dm",
    members: Optional[List[str]] = None,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Optional[GeneratedMessage]:
    """
    Generate a message completion for the target person.

    Args:
        model: Trained model with LoRA adapter
        tokenizer: Tokenizer
        context_messages: Recent conversation history
        target_name: Who to generate for
        chat_type: "dm" or "group"
        members: Participant names
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        GeneratedMessage if successful, None otherwise
    """
    prompt = build_prompt(
        context_messages,
        target_name,
        chat_type=chat_type,
        members=members,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if "\n" in raw_text:
        raw_text = raw_text.split("\n")[0]

    return parse_generated(raw_text, target_name)


def show_interactive_help():
    """Display help panel for interactive mode."""
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="dim")

    table.add_row("/target <name>", "Set who to generate for")
    table.add_row("/members <n1> <n2> ...", "Set conversation members")
    table.add_row("/type <dm|group>", "Set chat type")
    table.add_row("/context", "Show conversation history")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/help", "Show this help")
    table.add_row("/quit", "Exit")
    table.add_row("", "")
    table.add_row("<name>: <message>", "Add message and generate response")

    console.print(table)


def show_status(target_name: str, members: List[str], chat_type: str, context_len: int):
    """Display current session status."""
    status = Text()
    status.append("Target: ", style="dim")
    status.append(target_name, style="bold cyan")
    status.append("  |  ", style="dim")
    status.append("Type: ", style="dim")
    status.append(chat_type, style="yellow")
    status.append("  |  ", style="dim")
    status.append("Members: ", style="dim")
    status.append(", ".join(members), style="white")
    status.append("  |  ", style="dim")
    status.append("Context: ", style="dim")
    status.append(f"{context_len} msgs", style="green" if context_len > 0 else "dim")
    console.print(status)


def interactive_mode(model, tokenizer):
    """Interactive CLI for testing the model."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Ventriloquist[/] Interactive Mode",
            border_style="cyan",
        )
    )
    console.print()
    show_interactive_help()
    console.print()

    context: List[Dict[str, str]] = []
    target_name = "Contact"
    members = ["User", "Contact"]
    chat_type = "dm"

    while True:
        try:
            show_status(target_name, members, chat_type, len(context))
            user_input = Prompt.ask("[bold]>[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            console.print("[dim]Goodbye![/]")
            break

        if user_input == "/help":
            show_interactive_help()
            continue

        if user_input == "/clear":
            context = []
            console.print("[yellow]Conversation cleared.[/]")
            continue

        if user_input == "/context":
            if not context:
                console.print("[dim]No messages in context.[/]")
            else:
                console.print(Panel("[bold]Conversation History[/]", border_style="dim"))
                for msg in context:
                    delta_style = "dim"
                    console.print(
                        f"  [bold]{msg['name']}[/] [{delta_style}]{msg['delta']}[/]: {msg['content']}"
                    )
            continue

        if user_input.startswith("/target "):
            target_name = user_input[8:].strip()
            if target_name not in members:
                members.append(target_name)
            console.print(f"[green]Target set to:[/] [bold]{target_name}[/]")
            continue

        if user_input.startswith("/members "):
            members = user_input[9:].strip().split()
            if target_name not in members:
                members.append(target_name)
            console.print(f"[green]Members set to:[/] {', '.join(members)}")
            continue

        if user_input.startswith("/type "):
            new_type = user_input[6:].strip()
            if new_type in ("dm", "group"):
                chat_type = new_type
                console.print(f"[green]Chat type set to:[/] [yellow]{chat_type}[/]")
            else:
                console.print("[red]Invalid type. Use 'dm' or 'group'.[/]")
            continue

        if user_input.startswith("/"):
            console.print(f"[red]Unknown command:[/] {user_input}")
            console.print("[dim]Type /help for available commands.[/]")
            continue

        if ": " in user_input:
            parts = user_input.split(": ", 1)
            name = parts[0]
            content = parts[1] if len(parts) > 1 else ""

            # Add sender to members if new
            if name not in members:
                members.append(name)

            context.append({
                "name": name,
                "delta": "<5m>",
                "content": content,
            })

            # Show the input message
            console.print(f"\n  [bold blue]{name}[/] [dim]<5m>[/]: {content}")

            with console.status(f"[cyan]Generating response from {target_name}...[/]"):
                response = generate_response(
                    model,
                    tokenizer,
                    context[-50:],
                    target_name,
                    chat_type=chat_type,
                    members=members,
                )

            if response:
                console.print(
                    f"  [bold green]{target_name}[/] [dim]{response.delta}[/]: {response.content}\n"
                )
                context.append({
                    "name": response.name,
                    "delta": response.delta,
                    "content": response.content,
                })
            else:
                console.print("[red]Failed to generate valid response.[/]\n")
        else:
            console.print("[red]Invalid format.[/] Use: [cyan]<name>: <message>[/]")


def main():
    parser = argparse.ArgumentParser(description="Ventriloquist inference")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/ventriloquist/final",
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Use 8-bit quantization (for 24GB GPUs)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization (for 16GB GPUs)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(
        Path(args.adapter_path),
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
