"""
Interactive inference for Ventriloquist.

Load trained model and generate message completions interactively.

Usage:
    >>> python -m src.inference.interactive
    >>> python -m src.inference.interactive --adapter_path checkpoints/my_model

File: inference/interactive.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-26
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

from .utils import get_device, load_model

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

    table.add_row("/sender <name>", "Set your name")
    table.add_row("/target <name>", "Set who to generate for")
    table.add_row("/members <n1>, <n2>, ...", "Set members (comma-separated)")
    table.add_row("/type <dm|group>", "Set chat type")
    table.add_row("/context", "Show conversation history")
    table.add_row("/clear", "Clear conversation history")
    table.add_row("/help", "Show this help")
    table.add_row("/quit", "Exit")
    table.add_row("", "")
    table.add_row("<message>", "Send as yourself, get response from target")

    console.print(table)


def show_status(sender_name: str, target_name: str, chat_type: str, context_len: int):
    """Display current session status."""
    status = Text()
    status.append("You: ", style="dim")
    status.append(sender_name, style="bold blue")
    status.append("  →  ", style="dim")
    status.append(target_name, style="bold cyan")
    status.append("  |  ", style="dim")
    status.append(chat_type, style="yellow")
    status.append("  |  ", style="dim")
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

    # Onboarding
    console.print("[bold]Setup[/]")
    sender_name = Prompt.ask("  Your name").strip()
    target_name = Prompt.ask("  Target name (who to generate for)").strip()
    members_input = Prompt.ask("  All members (comma-separated)", default=f"{sender_name}, {target_name}")
    members = [m.strip() for m in members_input.split(",") if m.strip()]

    if sender_name not in members:
        members.append(sender_name)
    if target_name not in members:
        members.append(target_name)

    chat_type = "dm" if len(members) == 2 else "group"

    console.print()
    show_interactive_help()
    console.print()

    context: List[Dict[str, str]] = []

    while True:
        try:
            show_status(sender_name, target_name, chat_type, len(context))
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

        if user_input.startswith("/sender "):
            sender_name = user_input[8:].strip()
            if sender_name not in members:
                members.append(sender_name)
            console.print(f"[green]Sender set to:[/] [bold]{sender_name}[/]")
            continue

        if user_input.startswith("/members "):
            members = [m.strip() for m in user_input[9:].split(",") if m.strip()]
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

        # Message input - use sender_name by default, or "Name: message" to override
        if ": " in user_input and user_input.split(": ", 1)[0] in members:
            parts = user_input.split(": ", 1)
            name = parts[0]
            content = parts[1]
        else:
            name = sender_name
            content = user_input

        context.append({
            "name": name,
            "delta": "<5m",
            "content": content,
        })

        # Show the input message
        console.print(f"\n  [bold blue]{name}[/] [dim]<5m[/]: {content}")

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


def auto_mode(
    model,
    tokenizer,
    members: List[str],
    chat_type: str,
    max_turns: int = 0,
    seed_message: Optional[str] = None,
    seed_sender: Optional[str] = None,
):
    """
    Auto-generate a conversation between members.

    Streams output to terminal, letting the model hallucinate naturally.
    """
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Ventriloquist[/] Auto Mode\n[dim]{chat_type} with {', '.join(members)}[/]",
            border_style="cyan",
        )
    )
    console.print("[dim]Press Ctrl+C to stop[/]\n")

    # Build initial prompt with header
    header = build_header(chat_type, members)
    prompt = header + "\n"

    # Add seed message if provided
    if seed_message:
        sender = seed_sender or members[0]
        seed_json = json.dumps({
            "name": sender,
            "delta": "<1m",
            "content": seed_message,
        }, ensure_ascii=False)
        prompt += seed_json + "\n"
        console.print(f"[bold blue]{sender}[/] [dim]<1m[/]: {seed_message}")

    turn_count = 0
    colors = ["blue", "green", "magenta", "yellow"]

    try:
        while max_turns == 0 or turn_count < max_turns:
            # Tokenize current prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate multiple messages at once
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Get only the new tokens
            new_tokens = outputs[0][inputs.input_ids.shape[1]:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            if not new_text.strip():
                console.print("[dim]Model stopped generating.[/]")
                break

            # Split on }{ to handle multiple JSON objects without newlines
            # Also split on newlines for properly formatted output
            raw_chunks = new_text.replace("}{", "}\n{").strip().split("\n")
            for line in raw_chunks:
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                    name = msg.get("name", "???")
                    delta = msg.get("delta", "")
                    content = msg.get("content", "")

                    # Color based on member index
                    member_idx = members.index(name) if name in members else 0
                    color = colors[member_idx % len(colors)]

                    console.print(f"[bold {color}]{name}[/] [dim]{delta}[/]: {content}")
                    turn_count += 1

                    # Add to prompt for context
                    prompt += line + "\n"

                    if max_turns > 0 and turn_count >= max_turns:
                        break
                except json.JSONDecodeError:
                    # Not valid JSON, might be partial - skip
                    console.print(f"[dim red]Parse error: {line[:400]}...[/]")
                    continue

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/]")

    console.print(f"\n[dim]Generated {turn_count} messages.[/]")


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
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-generate conversation (no user input)",
    )
    parser.add_argument(
        "--members",
        type=str,
        help="Comma-separated member names for auto mode",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=0,
        help="Max messages to generate in auto mode (0=unlimited)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        help="Initial message to start the conversation (auto mode)",
    )
    parser.add_argument(
        "--seed_sender",
        type=str,
        help="Who sends the seed message (defaults to first member)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(
        Path(args.adapter_path),
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    if args.auto:
        if not args.members:
            console.print("[red]--auto requires --members 'Name1, Name2'[/]")
            return
        members = [m.strip() for m in args.members.split(",")]
        chat_type = "dm" if len(members) == 2 else "group"
        auto_mode(
            model,
            tokenizer,
            members,
            chat_type,
            args.turns,
            seed_message=args.seed,
            seed_sender=args.seed_sender,
        )
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
