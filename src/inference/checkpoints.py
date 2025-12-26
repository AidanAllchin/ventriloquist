"""
Checkpoint discovery utilities.

Finds available trained model checkpoints.

File: inference/checkpoints.py
Author: Aidan Allchin
Created: 2025-12-26
Last Modified: 2025-12-26
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Checkpoint:
    """Information about a saved checkpoint."""

    path: Path
    name: str
    step: Optional[int]  # None for 'final'
    modified: datetime


def find_checkpoints(base_dir: Path = Path("checkpoints")) -> List[Checkpoint]:
    """
    Find all available checkpoints.

    Looks for directories containing adapter_config.json (PEFT adapters).

    Returns:
        List of Checkpoint objects sorted by modification time (newest first)
    """
    checkpoints = []

    if not base_dir.exists():
        return checkpoints

    # Look for adapter_config.json recursively
    for adapter_config in base_dir.rglob("adapter_config.json"):
        checkpoint_dir = adapter_config.parent

        # Parse checkpoint name
        name = checkpoint_dir.name
        parent_name = checkpoint_dir.parent.name

        # Determine step number
        step = None
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
            except (IndexError, ValueError):
                pass

        # Build display name
        if parent_name != base_dir.name:
            display_name = f"{parent_name}/{name}"
        else:
            display_name = name

        checkpoints.append(
            Checkpoint(
                path=checkpoint_dir,
                name=display_name,
                step=step,
                modified=datetime.fromtimestamp(adapter_config.stat().st_mtime),
            )
        )

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda c: c.modified, reverse=True)

    return checkpoints


def select_checkpoint(checkpoints: List[Checkpoint]) -> Optional[Path]:
    """
    Display checkpoints and let user select one.

    Returns:
        Path to selected checkpoint, or None if cancelled
    """
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt
    from rich import box

    console = Console()

    if not checkpoints:
        console.print("[red]No checkpoints found in checkpoints/[/]")
        return None

    console.print("\n[bold]Available Checkpoints[/]\n")

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Step", style="yellow", justify="right")
    table.add_column("Modified", style="dim")

    for i, cp in enumerate(checkpoints, 1):
        step_str = str(cp.step) if cp.step else "final"
        modified_str = cp.modified.strftime("%Y-%m-%d %H:%M")
        table.add_row(str(i), cp.name, step_str, modified_str)

    console.print(table)
    console.print()

    try:
        choice = Prompt.ask("Select checkpoint (or 'q' to quit)")
        if choice.lower() == "q":
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(checkpoints):
            return checkpoints[idx].path
        console.print("[red]Invalid selection[/]")
        return None
    except (ValueError, KeyboardInterrupt):
        return None
