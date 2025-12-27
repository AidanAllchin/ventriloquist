"""
Database operations for training data storage and retrieval.

File: database/training_data.py
Author: Aidan Allchin
Created: 2025-12-24
Last Modified: 2025-12-27
"""

import json
import logging
from typing import List, Optional

import aiosqlite

from .common import LOCAL_DB_PATH
from ..models import TrainingMessage

log = logging.getLogger(__name__)


async def store_training_messages(messages: List[TrainingMessage]) -> int:
    """
    Store training messages to the database.

    Args:
        messages: List of TrainingMessage objects to store

    Returns:
        Number of messages stored
    """
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        # Clear existing training messages
        await conn.execute("DELETE FROM training_messages")

        for msg in messages:
            await conn.execute(
                """
                INSERT INTO training_messages (
                    chat_id, from_contact, timestamp, content, content_type,
                    is_group_chat, chat_members, reply_to_text, thread_originator_guid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.chat_id,
                    msg.from_contact,
                    msg.timestamp,
                    msg.content,
                    msg.content_type,
                    1 if msg.is_group_chat else 0,
                    json.dumps(msg.chat_members),
                    msg.reply_to_text,
                    msg.thread_originator_guid,
                ),
            )

        await conn.commit()

    log.info(f"Stored {len(messages)} training messages to database")
    return len(messages)

async def get_training_messages_by_chat(chat_id: str) -> List[TrainingMessage]:
    """
    Get all training messages for a specific chat, ordered by timestamp.

    Args:
        chat_id: The chat identifier

    Returns:
        List of TrainingMessage objects
    """
    messages = []

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(
            """
            SELECT chat_id, from_contact, timestamp, content, content_type,
                   is_group_chat, chat_members, reply_to_text, thread_originator_guid
            FROM training_messages
            WHERE chat_id = ?
            ORDER BY timestamp
            """,
            (chat_id,),
        ) as cursor:
            async for row in cursor:
                messages.append(
                    TrainingMessage(
                        chat_id=row[0],
                        from_contact=row[1],
                        timestamp=row[2],
                        content=row[3],
                        content_type=row[4],
                        is_group_chat=bool(row[5]),
                        chat_members=json.loads(row[6]),
                        reply_to_text=row[7],
                        thread_originator_guid=row[8],
                    )
                )

    return messages

async def get_all_chat_ids() -> List[str]:
    """
    Get all unique chat IDs from training messages.

    Returns:
        List of unique chat IDs
    """
    chat_ids = []

    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(
            "SELECT DISTINCT chat_id FROM training_messages ORDER BY chat_id"
        ) as cursor:
            async for row in cursor:
                chat_ids.append(row[0])

    return chat_ids

async def get_chat_metadata(chat_id: str) -> Optional[dict]:
    """
    Get metadata for a specific chat.

    Args:
        chat_id: The chat identifier

    Returns:
        Dict with chat_type, participants, message_count, first_ts, last_ts
    """
    async with aiosqlite.connect(LOCAL_DB_PATH) as conn:
        async with conn.execute(
            """
            SELECT is_group_chat, chat_members,
                   COUNT(*) as message_count,
                   MIN(timestamp) as first_ts,
                   MAX(timestamp) as last_ts
            FROM training_messages
            WHERE chat_id = ?
            GROUP BY chat_id
            """,
            (chat_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    "chat_type": "group" if row[0] else "dm",
                    "participants": json.loads(row[1]),
                    "message_count": row[2],
                    "first_ts": row[3],
                    "last_ts": row[4],
                }

    return None
