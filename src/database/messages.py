"""
File: database/messages.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from datetime import datetime
import logging
import json
from typing import Dict, List, Optional, Tuple

import aiosqlite

from .common import LOCAL_DB_PATH
from ..models import MessageRecord

log = logging.getLogger(__name__)


async def get_last_synced_timestamp() -> Tuple[Optional[datetime], Optional[str]]:
    """Get the timestamp and guid of the last synced message from local database"""
    try:
        async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
            async with conn.execute(
                """SELECT timestamp, guid FROM text_messages
                    ORDER BY timestamp DESC LIMIT 1""",
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return datetime.fromisoformat(row[0]), row[1]
        return None, None
    except Exception as e:
        log.error(f"Error getting last synced timestamp: {e}")
        return None, None

async def sync_batch_to_local_db(batch: List[MessageRecord]) -> int:
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        for msg in batch:
            await conn.execute("""
                INSERT INTO text_messages (
                    message_id, guid, text, timestamp, sender_id,
                    recipient_id, is_from_me, service, chat_identifier,
                    is_group_chat, group_chat_name, group_chat_participants,
                    has_attachments, is_read, read_timestamp, delivered_timestamp,
                    reply_to_guid, thread_originator_guid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(guid) DO UPDATE SET
                    text = excluded.text,
                    is_read = excluded.is_read,
                    read_timestamp = excluded.read_timestamp,
                    delivered_timestamp = excluded.delivered_timestamp
            """, (
                msg.message_id,
                msg.guid,
                msg.text,
                msg.timestamp.isoformat(),
                msg.sender_id,
                msg.recipient_id,
                1 if msg.is_from_me else 0,
                msg.service,
                msg.chat_identifier,
                1 if msg.is_group_chat else 0,
                msg.group_chat_name,
                json.dumps(msg.group_chat_participants) if msg.group_chat_participants else None,
                1 if msg.has_attachments else 0,
                1 if msg.is_read else 0,
                msg.date_read.isoformat() if msg.date_read else None,
                msg.date_delivered.isoformat() if msg.date_delivered else None,
                msg.reply_to_guid,
                msg.thread_originator_guid
            ))
        await conn.commit()
    return len(batch)

async def get_messages_with_identifiers(identifiers: List[str]) -> List[MessageRecord]:
    """
    Get all individual (1:1) messages with the given identifiers.

    Args:
        identifiers: List of phone numbers/emails to query

    Returns:
        List of MessageRecord objects for individual chats with these identifiers
    """
    if not identifiers:
        return []

    messages = []

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Query for messages where sender_id OR recipient_id is in identifiers
        # and is_group_chat = 0
        placeholders = ",".join("?" * len(identifiers))
        query = f"""
            SELECT * FROM text_messages
            WHERE is_group_chat = 0
            AND (sender_id IN ({placeholders}) OR recipient_id IN ({placeholders}))
            ORDER BY timestamp ASC
        """

        # Double the identifiers list for the two IN clauses
        params = identifiers + identifiers

        async with conn.execute(query, params) as cursor:
            columns = [description[0] for description in cursor.description]
            async for row in cursor:
                row_dict = dict(zip(columns, row))
                messages.append(MessageRecord.from_db_dict(row_dict))

    log.info(f"Retrieved {len(messages)} individual messages for {len(identifiers)} identifiers")
    return messages

async def get_messages_by_chat_identifiers(chat_identifiers: List[str]) -> List[MessageRecord]:
    """
    Get all messages for the given chat identifiers (typically group chats).

    Args:
        chat_identifiers: List of chat_identifier values to query

    Returns:
        List of MessageRecord objects for these chats
    """
    if not chat_identifiers:
        return []

    messages = []

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        placeholders = ",".join("?" * len(chat_identifiers))
        query = f"""
            SELECT * FROM text_messages
            WHERE chat_identifier IN ({placeholders})
            ORDER BY timestamp ASC
        """

        async with conn.execute(query, chat_identifiers) as cursor:
            columns = [description[0] for description in cursor.description]
            async for row in cursor:
                row_dict = dict(zip(columns, row))
                messages.append(MessageRecord.from_db_dict(row_dict))

    log.info(f"Retrieved {len(messages)} messages for {len(chat_identifiers)} chat identifiers")
    return messages

async def get_message_texts_by_guids(guids: List[str], batch_size: int = 500) -> Dict[str, str]:
    """
    Get message text content for a list of GUIDs.

    Args:
        guids: List of message GUIDs to look up
        batch_size: Number of GUIDs to query at once (SQLite has variable limits)

    Returns:
        Dictionary mapping GUID -> message text (only for messages with text)
    """
    if not guids:
        return {}

    result = {}

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Process in batches to avoid SQLite variable limits
        for i in range(0, len(guids), batch_size):
            batch = guids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            query = f"""
                SELECT guid, text FROM text_messages
                WHERE guid IN ({placeholders})
                AND text IS NOT NULL
            """

            async with conn.execute(query, batch) as cursor:
                async for row in cursor:
                    guid, text = row
                    if text:
                        result[guid] = text

    return result

async def detect_user_identifiers_from_db(exclude_identifiers: set[str] = None) -> set[str]:
    """
    Detect user identifiers from the database.

    Strategy: Look at recipient_id from individual messages where is_from_me = 0.
    When someone sends YOU a message, recipient_id is YOUR identifier.

    Args:
        exclude_identifiers: Set of identifiers to exclude (e.g., contact identifiers)

    Returns:
        Set of detected user identifiers
    """
    if exclude_identifiers is None:
        exclude_identifiers = set()

    user_identifiers = set()

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        query = """
            SELECT DISTINCT recipient_id, COUNT(*) as count
            FROM text_messages
            WHERE recipient_id IS NOT NULL
            AND is_from_me = 0
            AND is_group_chat = 0
            GROUP BY recipient_id
            ORDER BY count DESC
            LIMIT 5
        """

        async with conn.execute(query) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                identifier = row[0]
                count = row[1]
                # Only include if it's not in exclude list
                if identifier and identifier not in exclude_identifiers:
                    user_identifiers.add(identifier)
                    log.info(
                        f"  Found user identifier: {identifier} "
                        f"(appeared in {count} received messages)"
                    )

    return user_identifiers
