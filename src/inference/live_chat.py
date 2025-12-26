"""
Live chat mode with real iMessage context.

Fetches actual conversation history from iMessage and allows
interactive chat with model predictions.

File: inference/live_chat.py
Author: Aidan Allchin
Created: 2025-12-26
Last Modified: 2025-12-26
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from .imessage import (
    ChatInfo,
    RawMessage,
    check_imessage_access,
    fetch_recent_messages,
    list_chats,
)
from .formatting import (
    compute_delta,
    format_message,
    messages_to_prompt,
)
from .utils import load_model

load_dotenv()

console = Console()
MY_NAME = os.getenv("MY_NAME", "Me")
CONTEXT_SIZE = 49  # Model expects 49 context messages, predicts the 50th


def _select_chat(chats: List[ChatInfo]) -> Optional[ChatInfo]:
    """Display chat list and let user select one."""
    console.print("\n[bold]Available Chats[/]\n")

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="yellow", width=6)
    table.add_column("Messages", justify="right", style="green")
    table.add_column("Last Active", style="dim")

    for i, chat in enumerate(chats[:30], 1):  # Show top 30
        chat_type = "group" if chat.is_group else "dm"
        last_active = (
            chat.last_message_date.strftime("%Y-%m-%d")
            if chat.last_message_date
            else "unknown"
        )
        table.add_row(
            str(i),
            chat.display_name[:30],
            chat_type,
            str(chat.message_count),
            last_active,
        )

    console.print(table)
    console.print()

    try:
        choice = Prompt.ask("Select chat number (or 'q' to quit)")
        if choice.lower() == "q":
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(chats):
            return chats[idx]
        console.print("[red]Invalid selection[/]")
        return None
    except (ValueError, KeyboardInterrupt):
        return None


def _generate_responses(
    model,
    tokenizer,
    prompt: str,
    target_name: str,
    my_name: str,
    max_consecutive: int = 5,
    debug: bool = False,
) -> List[dict]:
    """
    Generate responses until the model predicts it's the user's turn.

    Stops when:
    - Model generates a message from my_name
    - Max consecutive messages reached
    - Generation fails

    Returns list of generated message dicts.
    """
    generated = []
    current_prompt = prompt

    if debug:
        console.print(Panel("[bold]Debug: Initial Prompt[/]", border_style="yellow"))
        console.print(f"[dim]{prompt}[/]")
        console.print()

    for _ in range(max_consecutive):
        # Add prefill to guide generation toward target
        prefill = f'{{"name": "{target_name}", "'
        full_prompt = current_prompt + "\n" + prefill

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if debug:
            console.print(Panel("[bold]Debug: Raw Model Output[/]", border_style="yellow"))
            console.print(f"[dim]Prefill:[/] {prefill}")
            console.print(f"[dim]Generated:[/] {raw_text}")
            console.print()

        # Reconstruct full JSON
        full_json = f'{{"name": "{target_name}", "{raw_text}'

        # Extract first complete JSON object
        try:
            # Find the closing brace
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
                break

            json_str = full_json[:end_pos]
            msg = json.loads(json_str)

            # Check if model is predicting user's turn
            if msg.get("name") == my_name:
                break

            generated.append(msg)
            current_prompt += "\n" + json.dumps(msg, ensure_ascii=False)

        except json.JSONDecodeError:
            break

    return generated


def _generate_continuation(
    model,
    tokenizer,
    prompt: str,
    num_messages: int = 5,
    stop_on_name: Optional[str] = None,
    debug: bool = False,
) -> List[dict]:
    """
    Generate a continuation of the conversation without forcing a specific speaker.

    The model decides who speaks next based on the conversation context.
    Generates in batches until we have enough messages or stop_on_name speaks.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Current conversation prompt
        num_messages: Number of messages to generate (0 = no limit, use stop_on_name)
        stop_on_name: Stop generating when this person speaks (e.g., MY_NAME)

    Returns:
        List of generated message dicts with name, delta, content.
    """
    generated = []
    current_prompt = prompt
    max_iterations = 10  # Safety limit

    if debug:
        console.print(Panel("[bold]Debug: Initial Prompt[/]", border_style="yellow"))
        console.print(f"[dim]{prompt}[/]")
        console.print()

    for iteration in range(max_iterations):
        if num_messages > 0 and len(generated) >= num_messages:
            break

        # Add prefill to prompt model to generate a new message
        prefill = '\n{"name": "'
        full_prompt = current_prompt + prefill

        if debug and iteration == 0:
            console.print(f"[dim]Prefill:[/] {prefill.strip()}")
            console.print()

        # Generate a larger chunk - enough for multiple messages
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

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

        new_tokens = outputs[0][inputs.input_ids.shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if debug:
            console.print(Panel(f"[bold]Debug: Raw Output (iteration {iteration + 1})[/]", border_style="yellow"))
            console.print(f"[dim]{raw_text}[/]")
            console.print()

        if not raw_text:
            break

        # Reconstruct first message from prefill + output
        first_msg = '{"name": "' + raw_text

        # Handle multiple JSON objects on same line
        first_msg = first_msg.replace("}{", "}\n{")
        lines = first_msg.split("\n")

        parsed_any = False
        should_stop = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)

                # Validate it has expected fields
                if "name" not in msg or "content" not in msg:
                    continue

                # Check if we should stop (model predicts user's turn)
                if stop_on_name and msg.get("name") == stop_on_name:
                    should_stop = True
                    break

                generated.append(msg)
                current_prompt += "\n" + json.dumps(msg, ensure_ascii=False)
                parsed_any = True

                if num_messages > 0 and len(generated) >= num_messages:
                    break

            except json.JSONDecodeError:
                continue

        if should_stop:
            break

        # If we couldn't parse anything this round, stop
        if not parsed_any:
            break

    return generated


def _show_help():
    """Display help for live chat commands."""
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="dim")

    table.add_row("/send", "Send your messages and get response")
    table.add_row("/continue [n]", "Generate n messages (default 5), model picks speakers")
    table.add_row("/target <name>", "Switch who to generate for (/target any to clear)")
    table.add_row("/clear", "Clear your pending messages")
    table.add_row("/context", "Show current conversation context")
    table.add_row("/debug", "Toggle debug mode (show full prompt and raw output)")
    table.add_row("/quit", "Exit live chat")
    table.add_row("/help", "Show this help")
    table.add_row("", "")
    table.add_row("<message>", "Type a message (queue until /send)")

    console.print(table)


def live_chat(adapter_path: Path):
    """
    Main live chat mode.

    Loads real iMessage context and allows interactive prediction.
    """
    # Check iMessage access
    if not check_imessage_access():
        console.print("[red]Cannot access iMessage database.[/]")
        console.print("[dim]Make sure you're on macOS and have granted Full Disk Access.[/]")
        return

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Ventriloquist[/] Live Chat",
            border_style="cyan",
        )
    )

    # Load model
    console.print("\n[dim]Loading model...[/]")
    model, tokenizer = load_model(adapter_path)
    console.print("[green]Model loaded.[/]")

    # List and select chat
    console.print("\n[dim]Fetching chats from iMessage...[/]")
    chats = list_chats(min_messages=20)

    if not chats:
        console.print("[red]No chats found with sufficient messages.[/]")
        return

    chat = _select_chat(chats)
    if chat is None:
        console.print("[dim]Goodbye![/]")
        return

    # Determine target
    # DM: always the other person
    # Group: None (model decides who responds)
    if chat.is_group:
        target_name: Optional[str] = None
    else:
        target_name = chat.display_name

    members = list(set(chat.members + [MY_NAME]))

    console.print()
    console.print(f"[bold]Chat:[/] {chat.display_name}")
    if chat.is_group:
        other_members = [m for m in members if m != MY_NAME]
        console.print(f"[bold]Members:[/] {', '.join(other_members)}")
        console.print(f"[bold]Target:[/] [cyan](any)[/] [dim]- use /target <name> to force[/]")
    else:
        console.print(f"[bold]Target:[/] [cyan]{target_name}[/]")
    console.print(f"[bold]You:[/] [blue]{MY_NAME}[/]")
    console.print()
    _show_help()
    console.print()

    # Fetch initial context (contact names resolved automatically)
    raw_messages = fetch_recent_messages(
        chat.chat_identifier,
        limit=CONTEXT_SIZE,
    )

    if not raw_messages:
        console.print("[yellow]No messages found in this chat.[/]")
        return

    console.print(f"[dim]Loaded {len(raw_messages)} messages of context.[/]\n")

    # Track conversation state
    context_messages = raw_messages.copy()
    pending_messages: List[str] = []
    debug_mode = False

    while True:
        # Show status
        pending_count = len(pending_messages)
        status = f"[dim]Context: {len(context_messages)} msgs"
        if pending_count > 0:
            status += f" | Pending: {pending_count} msgs"
        if debug_mode:
            status += " | [yellow]DEBUG[/]"
        status += "[/]"
        console.print(status)

        try:
            user_input = Prompt.ask("[bold]>[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue

        # Commands
        if user_input == "/quit":
            console.print("[dim]Goodbye![/]")
            break

        if user_input == "/help":
            _show_help()
            continue

        if user_input == "/clear":
            pending_messages = []
            console.print("[yellow]Pending messages cleared.[/]")
            continue

        if user_input == "/context":
            console.print(Panel("[bold]Current Context[/]", border_style="dim"))
            for msg in context_messages[-10:]:  # Show last 10
                style = "blue" if msg.sender == MY_NAME else "green"
                console.print(f"  [bold {style}]{msg.sender}[/]: {msg.text[:60]}...")
            if len(context_messages) > 10:
                console.print(f"  [dim]... and {len(context_messages) - 10} more[/]")
            continue

        if user_input == "/debug":
            debug_mode = not debug_mode
            if debug_mode:
                console.print("[yellow]Debug mode ON[/] - will show full prompt and raw output")
            else:
                console.print("[yellow]Debug mode OFF[/]")
            continue

        if user_input.startswith("/target "):
            new_target = user_input[8:].strip()
            if new_target.lower() in ("any", "none", "all"):
                if chat.is_group:
                    target_name = None
                    console.print("[green]Target cleared:[/] [cyan](any)[/] - model decides who responds")
                else:
                    console.print("[yellow]Can't clear target in DM - only one other person.[/]")
            elif new_target in members:
                target_name = new_target
                console.print(f"[green]Target set to:[/] [bold cyan]{target_name}[/]")
            else:
                console.print(f"[red]'{new_target}' not in members.[/]")
                console.print(f"[dim]Available: {', '.join(members)}[/]")
            continue

        if user_input == "/target":
            if target_name:
                console.print(f"[bold]Current target:[/] [cyan]{target_name}[/]")
            else:
                console.print(f"[bold]Current target:[/] [cyan](any)[/]")
            console.print(f"[dim]Members: {', '.join(members)}[/]")
            continue

        if user_input == "/continue" or user_input.startswith("/continue "):
            # Parse optional message count
            parts = user_input.split()
            num_messages = 5
            if len(parts) > 1:
                try:
                    num_messages = int(parts[1])
                except ValueError:
                    console.print("[red]Invalid number. Usage: /continue [n][/]")
                    continue

            # Build prompt from current context
            prompt = messages_to_prompt(
                context_messages,
                members,
                chat.is_group,
            )

            console.print()
            with console.status("[cyan]Generating...[/]"):
                responses = _generate_continuation(
                    model,
                    tokenizer,
                    prompt,
                    num_messages=num_messages,
                    debug=debug_mode,
                )

            if responses:
                colors = ["blue", "green", "magenta", "yellow", "cyan"]
                for resp in responses:
                    name = resp.get("name", "???")
                    delta = resp.get("delta", "")
                    content = resp.get("content", "")

                    # Color based on member
                    member_idx = members.index(name) if name in members else 0
                    color = colors[member_idx % len(colors)]

                    console.print(f"  [bold {color}]{name}[/] [dim]{delta}[/]: {content}")

                    # Add to context
                    context_messages.append(
                        RawMessage(
                            guid=f"gen-{datetime.now(timezone.utc).timestamp()}",
                            text=content,
                            timestamp=datetime.now(timezone.utc),
                            sender=name,
                            is_from_me=(name == MY_NAME),
                        )
                    )

                # Trim context
                while len(context_messages) > CONTEXT_SIZE:
                    context_messages.pop(0)
            else:
                console.print("[dim]No messages generated.[/]")

            console.print()
            continue

        if user_input == "/send":
            if not pending_messages:
                console.print("[yellow]No messages to send. Type some messages first.[/]")
                continue

            # Show what we're sending
            console.print()
            for msg_text in pending_messages:
                console.print(f"  [bold blue]{MY_NAME}[/] [dim]<1m[/]: {msg_text}")

            # Build prompt from current context
            prompt = messages_to_prompt(
                context_messages,
                members,
                chat.is_group,
            )

            # Add pending messages
            last_ts = context_messages[-1].timestamp if context_messages else None
            now = datetime.now(timezone.utc)

            for i, msg_text in enumerate(pending_messages):
                delta = compute_delta(now, last_ts) if i == 0 else "<1m"
                prompt += format_message(MY_NAME, delta, msg_text) + "\n"

                # Add to context as RawMessage
                context_messages.append(
                    RawMessage(
                        guid=f"user-{now.timestamp()}-{i}",
                        text=msg_text,
                        timestamp=now,
                        sender=MY_NAME,
                        is_from_me=True,
                    )
                )

            # Trim context from head to maintain size
            while len(context_messages) > CONTEXT_SIZE:
                context_messages.pop(0)

            pending_messages = []

            # Generate response
            console.print()
            if target_name:
                # DM or forced target: use prefill-guided generation
                with console.status(f"[cyan]{target_name} is typing...[/]"):
                    responses = _generate_responses(
                        model,
                        tokenizer,
                        prompt,
                        target_name,
                        MY_NAME,
                        debug=debug_mode,
                    )
            else:
                # Group chat: let model decide who responds, stop when it's my turn
                with console.status("[cyan]Generating...[/]"):
                    responses = _generate_continuation(
                        model,
                        tokenizer,
                        prompt,
                        num_messages=0,  # No limit
                        stop_on_name=MY_NAME,  # Stop when model predicts my turn
                        debug=debug_mode,
                    )

            if responses:
                colors = ["green", "magenta", "yellow", "cyan", "red"]
                for resp in responses:
                    name = resp.get("name", "???")
                    delta = resp.get("delta", "")
                    content = resp.get("content", "")

                    # Color based on member for groups, green for DMs
                    if target_name:
                        color = "green"
                    else:
                        other_members = [m for m in members if m != MY_NAME]
                        member_idx = other_members.index(name) if name in other_members else 0
                        color = colors[member_idx % len(colors)]

                    console.print(f"  [bold {color}]{name}[/] [dim]{delta}[/]: {content}")

                    # Add to context
                    context_messages.append(
                        RawMessage(
                            guid=f"gen-{datetime.now(timezone.utc).timestamp()}",
                            text=content,
                            timestamp=datetime.now(timezone.utc),
                            sender=name,
                            is_from_me=False,
                        )
                    )

                # Trim context
                while len(context_messages) > CONTEXT_SIZE:
                    context_messages.pop(0)
            else:
                console.print("[dim]No response generated.[/]")

            console.print()
            continue

        if user_input.startswith("/"):
            console.print(f"[red]Unknown command:[/] {user_input}")
            continue

        # Regular message - add to pending
        pending_messages.append(user_input)
        console.print(f"  [dim]+ {user_input}[/]")


def main():
    """CLI entry point for live chat."""
    import argparse

    parser = argparse.ArgumentParser(description="Ventriloquist Live Chat")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/ventriloquist/final",
        help="Path to trained LoRA adapter",
    )
    args = parser.parse_args()

    live_chat(Path(args.adapter_path))


if __name__ == "__main__":
    main()
