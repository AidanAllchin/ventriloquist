"""
Format messages into training prompt format.

Converts raw messages to the JSON-per-line format used for training.

File: inference/formatting.py
Author: Aidan Allchin
Created: 2025-12-26
Last Modified: 2025-12-27
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from .imessage import RawMessage


def compute_delta(current: datetime, previous: Optional[datetime]) -> str:
    """
    Compute time delta bucket between two timestamps.

    Buckets: <1m, <5m, <1h, <12h, <1d, 1d+
    """
    if previous is None:
        return "<1m"

    diff = current - previous
    seconds = diff.total_seconds()

    if seconds < 60:
        return "<1m"
    elif seconds < 300:
        return "<5m"
    elif seconds < 3600:
        return "<1h"
    elif seconds < 43200:
        return "<12h"
    elif seconds < 86400:
        return "<1d"
    else:
        return "1d+"


def build_header(chat_type: str, members: List[str], start_date: Optional[str] = None) -> str:
    """Build the window header JSON."""
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    header = {
        "type": chat_type,
        "members": sorted(members),
        "start": start_date,
    }
    return json.dumps(header, ensure_ascii=False)


def format_message(
    name: str,
    delta: str,
    text: str,
    content_type: str = "text"
) -> str:
    """Format a single message as JSON."""
    msg = {
        "name": name,
        "delta": delta,
        "content_type": content_type,
        "text": text,
    }
    return json.dumps(msg, ensure_ascii=False)


def messages_to_prompt(
    messages: List[RawMessage],
    members: List[str],
    is_group: bool,
) -> str:
    """
    Convert a list of messages to the full prompt format.

    Args:
        messages: List of RawMessage in chronological order
        members: List of member names
        is_group: Whether this is a group chat

    Returns:
        Full prompt string with header and message lines
    """
    if not messages:
        return ""

    chat_type = "group" if is_group else "dm"
    start_date = messages[0].timestamp.strftime("%Y-%m-%d")

    lines = [build_header(chat_type, members, start_date)]

    prev_timestamp = None
    for msg in messages:
        delta = compute_delta(msg.timestamp, prev_timestamp)
        lines.append(format_message(msg.sender, delta, msg.text))
        prev_timestamp = msg.timestamp

    # Each line ends with newline (matching training format)
    return "".join(line + "\n" for line in lines)


def append_user_messages(
    prompt: str,
    user_messages: List[str],
    sender_name: str,
    last_timestamp: Optional[datetime] = None,
) -> str:
    """
    Append user-typed messages to an existing prompt.

    Args:
        prompt: Existing prompt string (should already end with newline)
        user_messages: List of message contents from user
        sender_name: Name to use for the user
        last_timestamp: Timestamp of last message in prompt (for delta calculation)

    Returns:
        Updated prompt with new messages appended
    """
    result = prompt
    now = datetime.now()

    for i, content in enumerate(user_messages):
        if i == 0:
            delta = compute_delta(now, last_timestamp)
        else:
            delta = "<1m"  # Rapid-fire messages from user
        result += format_message(sender_name, delta, content) + "\n"

    return result
