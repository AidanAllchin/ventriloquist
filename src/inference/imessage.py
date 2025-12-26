"""
Direct iMessage database queries for live inference.

Provides read-only access to ~/Library/Messages/chat.db for fetching
real conversation context at inference time.

File: inference/imessage.py
Author: Aidan Allchin
Created: 2025-12-26
Last Modified: 2025-12-26
"""

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

IMESSAGE_DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
MY_NAME = os.getenv("MY_NAME", "Me")


def _load_identifier_mapping() -> Dict[str, str]:
    """Load identifier -> contact name mapping from contacts_to_ids.jsonl."""
    mapping = {}
    contacts_path = Path("data/contacts_to_ids.jsonl")

    if not contacts_path.exists():
        return mapping

    with open(contacts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                name = entry.get("contact", "")
                for identifier in entry.get("ids", []):
                    mapping[identifier] = name
            except json.JSONDecodeError:
                continue

    return mapping


@dataclass
class RawMessage:
    """A message from the iMessage database."""

    guid: str
    text: str
    timestamp: datetime
    sender: str  # Resolved to name
    is_from_me: bool


@dataclass
class ChatInfo:
    """Information about a chat."""

    chat_identifier: str
    display_name: str
    members: List[str]
    message_count: int
    last_message_date: Optional[datetime]
    is_group: bool


def _imessage_timestamp_to_datetime(timestamp: int) -> Optional[datetime]:
    """Convert iMessage timestamp (nanoseconds since 2001-01-01) to datetime."""
    if timestamp == 0:
        return None
    unix_timestamp = (timestamp / 1_000_000_000) + 978307200
    return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)


def _extract_text(text: Optional[str], attributed_body: Optional[bytes]) -> str:
    """Extract text from message, handling attributedBody if needed."""
    if text is not None and text.strip():
        return text

    if attributed_body is None:
        return ""

    try:
        bytes_text = attributed_body.split(b"NSString")[1][5:]
        if bytes_text[0] == 129:
            length = int.from_bytes(bytes_text[1:3], "little")
            bytes_text = bytes_text[3 : length + 3]
        else:
            length = bytes_text[0]
            bytes_text = bytes_text[1 : length + 1]
        return bytes_text.decode()
    except Exception:
        return ""


def check_imessage_access() -> bool:
    """Check if iMessage database is accessible."""
    return os.path.exists(IMESSAGE_DB_PATH)


def list_chats(min_messages: int = 10, only_known_contacts: bool = True) -> List[ChatInfo]:
    """
    List all chats with at least min_messages.

    Args:
        min_messages: Minimum message count to include
        only_known_contacts: If True, only show chats with contacts in contacts_to_ids.jsonl

    Returns:
        Chats sorted by message count descending.
    """
    # Load contact mapping
    id_to_name = _load_identifier_mapping() if only_known_contacts else {}
    known_identifiers = set(id_to_name.keys())

    query = """
    SELECT
        chat.chat_identifier,
        chat.display_name,
        chat.style,
        COUNT(DISTINCT message.ROWID) as message_count,
        MAX(message.date) as last_message_date
    FROM chat
    JOIN chat_message_join ON chat.ROWID = chat_message_join.chat_id
    JOIN message ON chat_message_join.message_id = message.ROWID
    GROUP BY chat.ROWID
    HAVING message_count >= ?
    ORDER BY message_count DESC
    """

    chats = []
    with sqlite3.connect(IMESSAGE_DB_PATH) as conn:
        cursor = conn.execute(query, (min_messages,))
        rows = cursor.fetchall()

        for row in rows:
            chat_identifier, display_name, style, count, last_date = row
            is_group = style == 43

            # Get members for this chat
            raw_members = _get_chat_members(conn, chat_identifier)

            # For DMs, the chat_identifier IS the contact identifier
            if not is_group:
                # Use chat_identifier as the member for DMs
                dm_identifier = chat_identifier

                # Filter: skip if not a known contact
                if only_known_contacts and dm_identifier not in known_identifiers:
                    continue

                # Use contact name for display
                display_name = id_to_name.get(dm_identifier, dm_identifier)
                members = [display_name]
            else:
                # Group chat - map member identifiers to names
                members = [id_to_name.get(m, m) for m in raw_members]

                # Keep group chat names as-is
                if not display_name:
                    display_name = f"Group ({len(members)} members)"

            if not display_name:
                display_name = "Unknown"

            if not members:
                members = ["Unknown"]

            chats.append(
                ChatInfo(
                    chat_identifier=chat_identifier,
                    display_name=display_name,
                    members=members, # type: ignore
                    message_count=count,
                    last_message_date=_imessage_timestamp_to_datetime(last_date)
                    if last_date
                    else None,
                    is_group=is_group,
                )
            )

    return chats


def _get_chat_members(conn: sqlite3.Connection, chat_identifier: str) -> List[str]:
    """Get member identifiers for a chat."""
    query = """
    SELECT DISTINCT handle.id
    FROM chat_handle_join
    JOIN handle ON chat_handle_join.handle_id = handle.ROWID
    JOIN chat ON chat_handle_join.chat_id = chat.ROWID
    WHERE chat.chat_identifier = ?
    ORDER BY handle.id
    """
    cursor = conn.execute(query, (chat_identifier,))
    return [row[0] for row in cursor.fetchall() if row[0]]


def fetch_recent_messages(
    chat_identifier: str,
    limit: int = 50,
    contact_name_map: Optional[dict] = None,
) -> List[RawMessage]:
    """
    Fetch the most recent messages from a chat.

    Args:
        chat_identifier: The chat to fetch from
        limit: Maximum number of messages to return
        contact_name_map: Optional mapping of identifier -> display name
                         (if None, loads from contacts_to_ids.jsonl)

    Returns:
        List of RawMessage in chronological order (oldest first)
    """
    if contact_name_map is None:
        contact_name_map = _load_identifier_mapping()

    query = """
    SELECT
        message.guid,
        message.text,
        message.attributedBody,
        message.date,
        handle.id as sender_id,
        message.is_from_me
    FROM message
    JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
    JOIN chat ON chat_message_join.chat_id = chat.ROWID
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    WHERE chat.chat_identifier = ?
    ORDER BY message.date DESC
    LIMIT ?
    """

    messages = []
    with sqlite3.connect(IMESSAGE_DB_PATH) as conn:
        cursor = conn.execute(query, (chat_identifier, limit))
        rows = cursor.fetchall()

        for row in rows:
            guid, text, attributed_body, date, sender_id, is_from_me = row

            # Extract text content
            content = _extract_text(text, attributed_body)
            if not content:
                continue

            # Resolve sender name
            if is_from_me:
                sender = MY_NAME
            else:
                sender = contact_name_map.get(sender_id, sender_id or "Unknown")

            timestamp = _imessage_timestamp_to_datetime(date)
            if timestamp is None:
                continue

            messages.append(
                RawMessage(
                    guid=guid,
                    text=content,
                    timestamp=timestamp,
                    sender=sender,
                    is_from_me=bool(is_from_me),
                )
            )

    # Return in chronological order (oldest first)
    return list(reversed(messages))
