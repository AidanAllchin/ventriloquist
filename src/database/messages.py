"""
File: database/messages.py
Author: Aidan Allchin
Created: 2025-12-23
Last Modified: 2025-12-23
"""

from datetime import datetime
import logging
import json
from typing import List, Optional, Tuple

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

