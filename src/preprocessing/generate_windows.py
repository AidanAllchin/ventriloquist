"""
Generate training windows from training messages using session-based windowing.

File: preprocessing/generate_windows.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-24
"""

from datetime import datetime, timedelta
import json
import logging
import os
from typing import List, Optional

import aiosqlite
from dotenv import load_dotenv

from ..database.common import LOCAL_DB_PATH
from ..models import TrainingMessage

log = logging.getLogger(__name__)

load_dotenv()
MY_NAME = os.getenv("MY_NAME")
if not MY_NAME:
    raise ValueError("MY_NAME not found in .env")

# Session boundary threshold (2 hours)
SESSION_GAP_THRESHOLD = timedelta(hours=2)

# Maximum messages per window (sanity cap)
MAX_MESSAGES_PER_WINDOW = 200

# Minimum context messages - keep pulling from previous sessions until we hit this
MIN_CONTEXT_MESSAGES = 25

# Time gap thresholds for markers (must be >= SESSION_GAP_THRESHOLD)
TIME_GAP_THRESHOLDS = [
    (timedelta(weeks=4), "--- long time later ---"),
    (timedelta(weeks=1), "--- weeks later ---"),
    (timedelta(days=2), "--- days later ---"),
    (timedelta(hours=12), "--- next day ---"),
    (timedelta(hours=2), "--- hours later ---"),
]


def get_time_gap_marker(gap: timedelta) -> str:
    """
    Get the appropriate time gap marker for a given duration.

    Args:
        gap: Time difference between messages

    Returns:
        Marker string like "--- hours later ---" or empty string if gap is small
    """
    for threshold, marker in TIME_GAP_THRESHOLDS:
        if gap >= threshold:
            return marker
    return ""


def identify_sessions(messages: List[TrainingMessage]) -> List[List[TrainingMessage]]:
    """
    Split messages into sessions based on time gaps.

    A new session starts when there's a gap of SESSION_GAP_THRESHOLD or more
    between consecutive messages.

    Args:
        messages: List of TrainingMessage objects, ordered by timestamp

    Returns:
        List of sessions, where each session is a list of messages
    """
    if not messages:
        return []

    sessions = []
    current_session = [messages[0]]

    for i in range(1, len(messages)):
        prev_ts = datetime.fromisoformat(messages[i - 1].timestamp)
        curr_ts = datetime.fromisoformat(messages[i].timestamp)
        gap = curr_ts - prev_ts

        if gap >= SESSION_GAP_THRESHOLD:
            # Start a new session
            sessions.append(current_session)
            current_session = [messages[i]]
        else:
            current_session.append(messages[i])

    # Don't forget the last session
    if current_session:
        sessions.append(current_session)

    return sessions


def format_header(messages: List[TrainingMessage]) -> str:
    """
    Format the conversation header.

    Args:
        messages: Messages in the conversation (to get metadata)

    Returns:
        Formatted header string
    """
    if not messages:
        return ""

    first_msg = messages[0]
    participants = ", ".join(first_msg.chat_members)

    if first_msg.is_group_chat:
        # For group chats, we don't have a display name in TrainingMessage
        # Use "Group" as placeholder - could be enhanced later
        return f"Group | {participants}"
    else:
        return f"DM | {participants}"


def format_message(msg: TrainingMessage) -> str:
    """
    Format a single message for the transcript.

    Args:
        msg: TrainingMessage to format

    Returns:
        Formatted message string
    """
    if msg.reply_to_text:
        return f'{msg.from_contact}: [replying to "{msg.reply_to_text}"] {msg.content}'
    else:
        return f"{msg.from_contact}: {msg.content}"


def render_transcript(
    current_session: List[TrainingMessage],
    context_sessions: Optional[List[List[TrainingMessage]]] = None,
) -> str:
    """
    Render a full transcript from messages with time gap markers between sessions.

    Args:
        current_session: Current session messages
        context_sessions: Optional list of previous sessions for context (ordered chronologically)

    Returns:
        Fully rendered transcript string
    """
    if not current_session and not context_sessions:
        raise ValueError("No messages or context sessions provided")

    # Get header from first available message
    if context_sessions and context_sessions[0]:
        header = format_header(context_sessions[0])
    else:
        header = format_header(current_session)

    lines = [header, "---"]

    # Track the last message timestamp for gap calculation
    last_msg_ts: Optional[datetime] = None

    # Add context from previous sessions with time markers between them
    if context_sessions:
        for session in context_sessions:
            if not session:
                continue

            # Add time gap marker if there's a previous session
            if last_msg_ts is not None:
                first_ts = datetime.fromisoformat(session[0].timestamp)
                gap = first_ts - last_msg_ts
                marker = get_time_gap_marker(gap)
                if marker:
                    lines.append(marker)

            # Add messages from this session
            for msg in session:
                lines.append(format_message(msg))

            # Update last message timestamp
            last_msg_ts = datetime.fromisoformat(session[-1].timestamp)

    # Add time marker before current session if we have context
    if last_msg_ts is not None and current_session:
        first_ts = datetime.fromisoformat(current_session[0].timestamp)
        gap = first_ts - last_msg_ts
        marker = get_time_gap_marker(gap)
        if marker:
            lines.append(marker)

    # Add current session messages
    for msg in current_session:
        lines.append(format_message(msg))

    return "\n".join(lines)


def generate_windows_from_messages(chat_id: str, messages: List[TrainingMessage]) -> List[dict]:
    """
    Generate training windows from a list of messages.

    Uses minimum context guarantee: each window will have at least MIN_CONTEXT_MESSAGES
    total messages (current session + context from previous sessions), unless the
    conversation doesn't have that many messages yet.

    Args:
        chat_id: The chat identifier
        messages: List of TrainingMessage objects, ordered by timestamp

    Returns:
        List of window dicts ready for DB insertion
    """
    if not messages:
        return []

    windows = []

    # Get metadata
    first_msg = messages[0]
    chat_type = "group" if first_msg.is_group_chat else "dm"
    participants = first_msg.chat_members

    # Split into sessions
    sessions = identify_sessions(messages)

    # Track all previous sessions for context building
    previous_sessions: List[List[TrainingMessage]] = []

    # Generate a window for each session
    for session in sessions:
        # Cap session size
        if len(session) > MAX_MESSAGES_PER_WINDOW:
            session = session[-MAX_MESSAGES_PER_WINDOW:]

        # Build context from previous sessions to meet minimum
        # Keep as list of sessions to preserve boundaries for time gap markers
        context_sessions: List[List[TrainingMessage]] = []
        context_message_count = 0
        messages_needed = max(0, MIN_CONTEXT_MESSAGES - len(session))

        if messages_needed > 0 and previous_sessions:
            # Pull from previous sessions (most recent first) until we have enough
            for prev_session in reversed(previous_sessions):
                if context_message_count >= messages_needed:
                    break
                # How many more do we need?
                still_needed = messages_needed - context_message_count
                # Take from the end of this previous session
                to_take = min(still_needed, len(prev_session))
                context_sessions.insert(0, prev_session[-to_take:])
                context_message_count += to_take

        # Cap total window size
        total_messages = context_message_count + len(session)
        if total_messages > MAX_MESSAGES_PER_WINDOW:
            # Trim context from the oldest sessions first
            excess = total_messages - MAX_MESSAGES_PER_WINDOW
            while excess > 0 and context_sessions:
                oldest_session = context_sessions[0]
                if len(oldest_session) <= excess:
                    # Remove entire oldest session
                    excess -= len(oldest_session)
                    context_sessions.pop(0)
                else:
                    # Trim the oldest session
                    context_sessions[0] = oldest_session[excess:]
                    excess = 0

        # Render transcript
        transcript = render_transcript(session, context_sessions if context_sessions else None)

        # Create window record
        window = {
            "chat_id": chat_id,
            "chat_type": chat_type,
            "chat_name": None,  # Could be enhanced to include group name
            "participants": json.dumps(participants),
            "transcript": transcript,
            "message_count": len(session),
            "session_start": session[0].timestamp,
            "session_end": session[-1].timestamp,
        }
        windows.append(window)

        # Add this session to previous sessions for future windows
        previous_sessions.append(session)

    return windows


async def generate_all_training_windows() -> int:
    """
    Generate training windows for all chats and store in database.

    Returns:
        Total number of windows generated
    """
    total_windows = 0

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Clear existing windows
        await conn.execute("DELETE FROM training_windows")

        # Get all unique chat IDs
        async with conn.execute(
            "SELECT DISTINCT chat_id FROM training_messages"
        ) as cursor:
            chat_ids = [row[0] async for row in cursor]

        log.info(f"Generating windows for {len(chat_ids)} conversations...")

        for i, chat_id in enumerate(chat_ids):
            # Get all messages for this chat
            async with conn.execute(
                """
                SELECT chat_id, from_contact, timestamp, content,
                       is_group_chat, chat_members, reply_to_text, thread_originator_guid
                FROM training_messages
                WHERE chat_id = ?
                ORDER BY timestamp
                """,
                (chat_id,),
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

            # Generate windows for this chat
            windows = generate_windows_from_messages(chat_id, messages)

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
                log.info(f"  Processed {i + 1}/{len(chat_ids)} chats...")

        await conn.commit()

    log.info(f"Generated {total_windows} training windows")
    return total_windows


async def export_windows_to_jsonl(output_path: str = "data/training_windows.jsonl") -> int:
    """
    Export training windows to JSONL file.

    Each line contains: {"window_id": int, "transcript": str}

    Args:
        output_path: Path to output file

    Returns:
        Number of windows exported
    """
    count = 0

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        with open(output_path, "w") as f:
            async with conn.execute(
                "SELECT window_id, transcript FROM training_windows"
            ) as cursor:
                async for row in cursor:
                    json.dump({"window_id": row[0], "transcript": row[1]}, f)
                    f.write("\n")
                    count += 1

    log.info(f"Exported {count} windows to {output_path}")
    return count
