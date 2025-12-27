"""
Main entry point for Ventriloquist.

Interactive CLI for running pipeline steps individually or together.

File: main.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-26
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

console = Console()

load_dotenv()

# Pipeline step definitions
STEPS = {
    "1": {
        "name": "Collect iMessage data",
        "description": "Extract messages from macOS iMessage database",
        "requires": "macOS",
    },
    "1.5": {
        "name": "Describe attachments",
        "description": "Generate descriptions for images/videos/audio via Gemini",
        "requires": "GEMINI_API_KEY",
    },
    "2": {
        "name": "Preprocess data",
        "description": "Create training messages, windows, and export to JSONL",
        "requires": None,
    },
    "3": {
        "name": "Train model",
        "description": "Fine-tune Qwen3-8B-Base with LoRA",
        "requires": "GPU (Linux/Windows recommended)",
    },
    "4": {
        "name": "Inference",
        "description": "Run trained model (interactive, auto, or live chat)",
        "requires": "Trained model",
    },
}

# Inference submenu options
INFERENCE_OPTIONS = {
    "1": {
        "name": "Interactive",
        "description": "Chat with synthetic context",
    },
    "2": {
        "name": "Auto",
        "description": "Watch model generate a conversation",
    },
    "3": {
        "name": "Live Chat",
        "description": "Chat with real iMessage context (macOS)",
    },
}


def show_menu():
    """Display the main menu."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Ventriloquist[/] - Personality Modeling via iMessage Fine-Tuning",
            border_style="cyan",
        )
    )
    console.print()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Step", style="cyan", width=4)
    table.add_column("Name", style="white")
    table.add_column("Description", style="dim")
    table.add_column("Requires", style="yellow")

    for key, step in STEPS.items():
        requires = step["requires"] or "-"
        table.add_row(key, step["name"], step["description"], requires)

    console.print(table)
    console.print()
    console.print("[dim]Commands:[/]")
    console.print("  [cyan]1, 1.5, 2, 3[/]  Run a pipeline step")
    console.print("  [cyan]4[/]             Inference (interactive, auto, or live chat)")
    console.print("  [cyan]a[/]             Run all pipeline steps (1, 1.5, 2, 3)")
    console.print("  [cyan]q[/]             Quit")
    console.print()

async def _run_step_1():
    """Collect iMessage data."""
    from src.collection import collect_data
    from src.database import init_local_database

    console.print("[dim]Initializing database...[/]")
    await init_local_database()

    console.print("[dim]Collecting messages from iMessage...[/]")
    await collect_data()

    console.print("[green]Collection complete![/]")


async def _run_step_1_5():
    """Generate descriptions for attachments using Gemini."""
    from src.gemini import process_attachments
    from src.database import get_processing_stats

    # Show current stats
    stats = await get_processing_stats()
    if stats["total_cached"] > 0:
        console.print(f"[dim]Already cached: {stats['total_cached']} attachments[/]")
        console.print(f"[dim]  With description: {stats['with_description']}[/]")
        console.print(f"[dim]  With error: {stats['with_error']}[/]")

    # Dry run first to show what will be processed
    console.print("[dim]Checking attachments to process...[/]")
    dry_stats = await process_attachments(dry_run=True)

    to_process = dry_stats.get("to_process", 0)
    if to_process == 0:
        console.print("[green]All attachments already cached![/]")
        return

    console.print(f"\n[bold]Attachments to process: {to_process}[/]")
    by_type = dry_stats.get("by_type", {})
    if isinstance(by_type, dict):
        for ctype, count in sorted(by_type.items()):
            console.print(f"  {ctype}: {count}")

    console.print()
    if not Confirm.ask(
        f"Process {to_process} attachments?",
        default=False,
    ):
        console.print("[dim]Skipped.[/]")
        return

    # Run for real
    console.print()
    await process_attachments(dry_run=False)


async def _run_step_2():
    """Preprocess data: create training messages, generate windows, export."""
    from src.preprocessing import (
        make_training_messages,
        generate_all_training_windows,
        export_windows_to_jsonl,
    )
    from src.database import store_training_messages

    console.print("[dim]Creating training messages...[/]")
    messages = await make_training_messages()
    console.print(f"  Created {len(messages):,} training messages")

    console.print("[dim]Storing to database...[/]")
    await store_training_messages(messages)

    console.print("[dim]Generating training windows...[/]")
    window_count = await generate_all_training_windows()
    console.print(f"  Generated {window_count:,} windows")

    console.print("[dim]Exporting to JSONL...[/]")
    await export_windows_to_jsonl()

    console.print(
        f"[green]Preprocessing complete![/] "
        f"{len(messages):,} messages → {window_count:,} windows"
    )

def _run_step_3():
    """Train the model."""
    import subprocess

    console.print("[bold]Starting training...[/]")
    console.print("[dim]Running: python -m src.training.train[/]")
    console.print()

    result = subprocess.run(
        [sys.executable, "-m", "src.training.train"],
        cwd=Path(__file__).parent,
    )

    if result.returncode != 0:
        console.print("[red]Training failed![/]")
    else:
        console.print("[green]Training complete![/]")


def _show_inference_menu():
    """Display the inference submenu."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Inference Options[/]",
            border_style="cyan",
        )
    )
    console.print()

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Option", style="cyan", width=6)
    table.add_column("Name", style="white")
    table.add_column("Description", style="dim")

    for key, opt in INFERENCE_OPTIONS.items():
        table.add_row(key, opt["name"], opt["description"])

    console.print(table)
    console.print()


def _run_step_4():
    """Inference submenu."""
    from src.inference import (
        live_chat,
        interactive_mode,
        auto_mode,
        load_model,
        find_checkpoints,
        select_checkpoint,
    )

    _show_inference_menu()

    choice = Prompt.ask(
        "Select option",
        choices=list(INFERENCE_OPTIONS.keys()) + ["q"],
        default="q",
    )

    if choice == "q":
        return

    # Select checkpoint first
    checkpoints = find_checkpoints()
    adapter_path = select_checkpoint(checkpoints)

    if adapter_path is None:
        console.print("[dim]Cancelled.[/]")
        return

    if choice == "1":
        # Interactive mode
        model, tokenizer = load_model(adapter_path)
        interactive_mode(model, tokenizer)
    elif choice == "2":
        # Auto mode - need members
        members_input = Prompt.ask("Members (comma-separated)")
        members = [m.strip() for m in members_input.split(",") if m.strip()]
        if len(members) < 2:
            console.print("[red]Need at least 2 members.[/]")
            return

        seed = Prompt.ask("Seed message (optional)", default="")
        turns = Prompt.ask("Max turns (0=unlimited)", default="0")

        model, tokenizer = load_model(adapter_path)
        chat_type = "dm" if len(members) == 2 else "group"
        auto_mode(
            model,
            tokenizer,
            members,
            chat_type,
            max_turns=int(turns),
            seed_message=seed if seed else None,
        )
    elif choice == "3":
        # Live chat
        live_chat(adapter_path)


async def run_all_steps():
    """Run all pipeline steps."""
    console.print("\n[bold cyan]Running full pipeline...[/]\n")

    console.rule("[bold]Step 1: Collect iMessage data")
    await _run_step_1()

    console.rule("[bold]Step 1.5: Describe attachments")
    await _run_step_1_5()

    console.rule("[bold]Step 2: Preprocess data")
    await _run_step_2()

    console.rule("[bold]Step 3: Train model")
    _run_step_3()

    console.print()
    console.print(Panel.fit("[bold green]Pipeline complete![/]", border_style="green"))

async def run_single_step(step: str):
    """Run a single pipeline step."""
    step_info = STEPS[step]
    console.rule(f"[bold]{step_info['name']}")

    if step == "1":
        await _run_step_1()
    elif step == "1.5":
        await _run_step_1_5()
    elif step == "2":
        await _run_step_2()
    elif step == "3":
        _run_step_3()
    elif step == "4":
        _run_step_4()

async def main():
    """Main entry point with interactive menu."""
    # Check for command-line argument for non-interactive use
    if len(sys.argv) > 1:
        step = sys.argv[1].lower()
        if step == "all" or step == "a":
            await run_all_steps()
            return
        elif step in STEPS:
            await run_single_step(step)
            return
        else:
            console.print(f"[red]Unknown step: {step}[/]")
            console.print("[dim]Valid steps: 1-3, a (all), q (quit)[/]")
            return

    # Interactive mode
    while True:
        show_menu()

        choice = Prompt.ask(
            "Select step",
            choices=list(STEPS.keys()) + ["a", "q"],
            default="q",
        )

        if choice == "q":
            console.print("[dim]Goodbye![/]")
            break
        elif choice == "a":
            await run_all_steps()
        else:
            await run_single_step(choice)

        console.print()
        if not Confirm.ask("Continue?", default=True):
            console.print("[dim]Goodbye![/]")
            break


if __name__ == "__main__":
    asyncio.run(main())
