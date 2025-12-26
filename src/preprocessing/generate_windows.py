"""
Generate training windows from training messages using sliding window approach.

Each message becomes the final message of its own training window, with up to
WINDOW_SIZE previous messages as context.

File: preprocessing/generate_windows.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-25
"""

from datetime import datetime, timedelta
import json
import logging
from typing import List

import aiosqlite

from ..database.common import LOCAL_DB_PATH
from ..models import TrainingMessage, compute_delta_bucket

log = logging.getLogger(__name__)

# Maximum messages per window (including the target message)
WINDOW_SIZE = 50


def format_window_header(messages: List[TrainingMessage]) -> str:
    """
    Format the JSON header for a training window.

    Args:
        messages: Messages in the window (to get metadata)

    Returns:
        JSON string: {"type": "dm"|"group", "members": [...], "start": "YYYY-MM-DD"}
    """
    if not messages:
        return ""

    first_msg = messages[0]
    chat_type = "group" if first_msg.is_group_chat else "dm"
    members = sorted(first_msg.chat_members)
    start_date = datetime.fromisoformat(first_msg.timestamp).strftime("%Y-%m-%d")

    return json.dumps(
        {"type": chat_type, "members": members, "start": start_date},
        ensure_ascii=False,
    )


def render_window(messages: List[TrainingMessage]) -> str:
    """
    Render a training window as header + JSON message lines.

    Args:
        messages: List of TrainingMessage objects in chronological order

    Returns:
        Full window transcript with header and message lines
    """
    if not messages:
        return ""

    lines = [format_window_header(messages)]

    # First message always gets "<1m" delta
    lines.append(messages[0].to_window_json("<1m"))

    # Subsequent messages get computed deltas
    for i in range(1, len(messages)):
        prev_ts = datetime.fromisoformat(messages[i - 1].timestamp)
        curr_ts = datetime.fromisoformat(messages[i].timestamp)
        delta = curr_ts - prev_ts

        # Handle negative deltas (shouldn't happen, but be safe)
        if delta < timedelta(0):
            delta = timedelta(0)

        bucket = compute_delta_bucket(delta)
        lines.append(messages[i].to_window_json(bucket))

    # Each line ends with newline (including last) so model learns to output newlines
    return "".join(line + "\n" for line in lines)


def generate_windows_from_messages(
    chat_id: str, messages: List[TrainingMessage]
) -> List[dict]:
    """
    Generate sliding training windows from a list of messages.

    Each message becomes the final message of its own window, with up to
    WINDOW_SIZE-1 previous messages as context.

    Args:
        chat_id: The chat identifier
        messages: List of TrainingMessage objects, ordered by timestamp

    Returns:
        List of window dicts ready for DB insertion
    """
    if not messages:
        return []

    windows = []
    first_msg = messages[0]
    chat_type = "group" if first_msg.is_group_chat else "dm"
    participants = sorted(first_msg.chat_members)

    # Generate a window for each message
    for i, target_msg in enumerate(messages):
        # Get context: up to WINDOW_SIZE messages ending with target
        start_idx = max(0, i - WINDOW_SIZE + 1)
        window_messages = messages[start_idx : i + 1]

        # Render the window
        transcript = render_window(window_messages)

        # Get timestamps for metadata
        window_start = datetime.fromisoformat(window_messages[0].timestamp)
        window_end = datetime.fromisoformat(target_msg.timestamp)

        window = {
            "chat_id": chat_id,
            "chat_type": chat_type,
            "chat_name": None,
            "participants": json.dumps(participants),
            "transcript": transcript,
            "message_count": len(window_messages),
            "session_start": window_start.isoformat(),
            "session_end": window_end.isoformat(),
        }
        windows.append(window)

    return windows


async def generate_all_training_windows() -> int:
    """
    Generate training windows for all conversations and store in database.

    Groups messages by participant set (not chat_id) since the same group
    of people may have multiple chat IDs over time.

    Returns:
        Total number of windows generated
    """
    total_windows = 0

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Clear existing windows
        await conn.execute("DELETE FROM training_windows")

        # Get all unique participant sets (same people = same conversation)
        async with conn.execute(
            "SELECT DISTINCT chat_members FROM training_messages"
        ) as cursor:
            participant_sets = [row[0] async for row in cursor]

        log.info(f"Generating windows for {len(participant_sets)} unique conversations...")

        for i, members_json in enumerate(participant_sets):
            # Get all messages for this participant set, across all chat IDs
            async with conn.execute(
                """
                SELECT chat_id, from_contact, timestamp, content,
                       is_group_chat, chat_members, reply_to_text, thread_originator_guid
                FROM training_messages
                WHERE chat_members = ?
                ORDER BY timestamp
                """,
                (members_json,),
            ) as cursor:
                messages = []
                async for row in cursor:
                    messages.append(
                        TrainingMessage(
                            chat_id=row[0],
                            from_contact=row[1],
                            timestamp=row[2],
                            content=row[3],
                            is_group_chat=bool(row[4]),
                            chat_members=json.loads(row[5]),
                            reply_to_text=row[6],
                            thread_originator_guid=row[7],
                        )
                    )

            # Use participant set as the conversation identifier
            windows = generate_windows_from_messages(members_json, messages)

            for window in windows:
                await conn.execute(
                    """
                    INSERT INTO training_windows (
                        chat_id, chat_type, chat_name, participants,
                        transcript, message_count, session_start, session_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        window["chat_id"],
                        window["chat_type"],
                        window["chat_name"],
                        window["participants"],
                        window["transcript"],
                        window["message_count"],
                        window["session_start"],
                        window["session_end"],
                    ),
                )

            total_windows += len(windows)

            if (i + 1) % 20 == 0:
                log.info(f"  Processed {i + 1}/{len(participant_sets)} conversations...")

        await conn.commit()

    log.info(f"Generated {total_windows} training windows")
    return total_windows


async def export_windows_to_jsonl(output_path: str = "data/training_windows.jsonl") -> int:
    """
    Export training windows to JSONL file.

    Each line contains: {"text": "<full transcript>"}

    Args:
        output_path: Path to output file

    Returns:
        Number of windows exported
    """
    count = 0

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        with open(output_path, "w") as f:
            async with conn.execute(
                "SELECT transcript FROM training_windows"
            ) as cursor:
                async for row in cursor:
                    json.dump({"text": row[0]}, f, ensure_ascii=False)
                    f.write("\n")
                    count += 1

    log.info(f"Exported {count} windows to {output_path}")
    return count
