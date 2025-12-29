#!/usr/bin/env python3
"""
Training Monitor Script

Polls W&B for the most recent training run and checks for issues.
If a problem is detected, invokes Claude to investigate and create an incident report.

Usage:
    python scripts/monitor_training.py              # One-shot check
    python scripts/monitor_training.py --daemon     # Run every 30 mins

Cron setup (alternative to --daemon):
    */30 * * * * cd ~/Desktop/Projects/ventriloquist && uv run python scripts/monitor_training.py
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import wandb
from dotenv import load_dotenv

def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    if not os.getenv("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not found in .env")
        sys.exit(1)
    
    if not os.getenv("WANDB_ENTITY"):
        print("ERROR: WANDB_ENTITY not found in .env")
        sys.exit(1)

load_env()

# Configuration
ENTITY = os.getenv("WANDB_ENTITY")
PROJECT = "ventriloquist"
CHECK_INTERVAL_MINS = 30

# Thresholds for detecting issues
HEARTBEAT_STALE_MINS = 15  # Alert if no heartbeat for this long
LOSS_SPIKE_THRESHOLD = 2.0  # Alert if loss exceeds this (way above normal ~0.6)
MIN_STEPS_FOR_LOSS_CHECK = 50  # Don't check loss until we have enough data


def get_latest_run() -> wandb.apis.public.Run | None:  # type: ignore[attr-defined]
    """Fetch the most recent run from the ventriloquist project."""
    api = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        order="-created_at",
        per_page=1
    )

    try:
        return next(iter(runs))
    except StopIteration:
        return None


def check_run_health(
    run: wandb.apis.public.Run  # type: ignore[attr-defined]
) -> tuple[bool, str | None]:
    """
    Check if a run is healthy.

    Returns:
        (is_healthy, issue_description)
    """
    # Check 1: Run state
    if run.state == "crashed":
        return False, f"Run crashed. State: {run.state}"

    if run.state == "failed":
        return False, f"Run failed. State: {run.state}"

    if run.state == "finished":
        # This might be expected - not necessarily an issue
        return True, None

    if run.state != "running":
        return False, f"Unexpected run state: {run.state}"

    # Check 2: Heartbeat freshness
    heartbeat = run.heartbeatAt
    if heartbeat:
        # Parse heartbeat time
        if isinstance(heartbeat, str):
            heartbeat_dt = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
        else:
            heartbeat_dt = heartbeat

        now = datetime.now(timezone.utc)
        stale_mins = (now - heartbeat_dt).total_seconds() / 60

        if stale_mins > HEARTBEAT_STALE_MINS:
            return False, f"Heartbeat stale for {stale_mins:.1f} minutes (threshold: {HEARTBEAT_STALE_MINS})"

    # Check 3: Loss sanity (NaN, explosion)
    summary = run.summary
    if summary:
        loss = summary.get("loss") or summary.get("train/loss")
        step = summary.get("_step", 0) or summary.get("train/global_step", 0)

        if loss is not None and step >= MIN_STEPS_FOR_LOSS_CHECK:
            # Check for NaN
            if loss != loss:  # NaN check
                return False, f"Loss is NaN at step {step}"

            # Check for explosion
            if loss > LOSS_SPIKE_THRESHOLD:
                return False, f"Loss exploded to {loss:.4f} at step {step} (threshold: {LOSS_SPIKE_THRESHOLD})"

    return True, None


def invoke_claude_investigation(run_id: str, issue: str):
    """Invoke Claude to investigate the issue and create an incident report."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    report_path = f"docs/incident_report_{timestamp}.md"

    prompt = f"""TRAINING ALERT: Issue detected with run {run_id}

Issue: {issue}

Please:
1. Query the W&B run to get detailed metrics and history
2. Analyze what went wrong
3. Create an incident report at {report_path} with:
   - Summary of the issue
   - Timeline of events (when did metrics start degrading?)
   - Possible causes
   - Recommended next steps
4. After creating the report, send a SHORT message (one sentence) to my phone via the send_message_to_device tool saying the training failed

Entity: {ENTITY}
Project: {PROJECT}
Run ID: {run_id}
"""

    print(f"[{datetime.now()}] Invoking Claude to investigate...")

    try:
        # Stream output in real-time while also capturing it
        process = subprocess.Popen(
            [
                "claude",
                "-p", prompt,
                "--dangerously-skip-permissions",  # Full autonomy, no prompts
            ],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )
        if not process or not process.stdout:
            print("ERROR: Failed to start Claude process")
            return

        # Stream output line by line
        output_lines = []
        for line in process.stdout:
            print(f"  [claude] {line}", end="")
            output_lines.append(line)

        process.wait(timeout=300)

        if process.returncode != 0:
            print(f"Claude exited with code {process.returncode}")
        else:
            print(f"[{datetime.now()}] Claude investigation complete. Check {report_path}")

    except subprocess.TimeoutExpired:
        if 'process' in locals():
            process.kill()  # type: ignore[attr-defined]
        else:
            print("ERROR: Process not found")
        print("Claude invocation timed out after 5 minutes")
    except FileNotFoundError:
        print("ERROR: 'claude' command not found. Is Claude Code installed?")


def monitor_once() -> bool:
    """
    Run a single monitoring check.

    Returns:
        True if healthy, False if issue detected
    """
    print(f"[{datetime.now()}] Checking training status...")

    run = get_latest_run()
    if not run:
        print("No runs found in project")
        return True  # Not an error condition

    print(f"  Run: {run.name} ({run.id})")
    print(f"  State: {run.state}")
    print(f"  Created: {run.createdAt}")

    is_healthy, issue = check_run_health(run)

    if is_healthy:
        summary = run.summary or {}
        loss = summary.get("loss") or summary.get("train/loss", "N/A")
        step = summary.get("_step", 0) or summary.get("train/global_step", 0)
        print(f"  Status: HEALTHY (loss={loss}, step={step})")
        return True
    else:
        print(f"  Status: ISSUE DETECTED - {issue}")
        if not issue:
            print("WARNING: No issue description provided")
            issue = "[No issue description provided. Investigate the run to determine the issue.]"
        invoke_claude_investigation(run.id, issue)
        return False


def run_daemon():
    """Run the monitor continuously."""
    print(f"Starting training monitor daemon (checking every {CHECK_INTERVAL_MINS} mins)")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            monitor_once()
            print(f"\nNext check in {CHECK_INTERVAL_MINS} minutes...\n")
            time.sleep(CHECK_INTERVAL_MINS * 60)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


def main():
    parser = argparse.ArgumentParser(description="Monitor W&B training runs")
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run continuously, checking every 30 minutes"
    )
    args = parser.parse_args()

    if args.daemon:
        run_daemon()
    else:
        healthy = monitor_once()
        sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
